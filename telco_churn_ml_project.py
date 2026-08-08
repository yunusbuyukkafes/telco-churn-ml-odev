"""
Uçtan Uca Makine Öğrenmesi Projesi — Telco Müşteri Ayrılma (Churn) Tahmini
Türkiye Yapay Zeka Akademisi Final Ödevi

Amaç:
    Telekomünikasyon müşterilerinin hizmeti bırakıp bırakmayacağını (Churn)
    sınıflandırma modelleriyle tahmin etmek. Veri inceleme, ön işleme,
    öznitelik mühendisliği, model eğitimi/karşılaştırma, çapraz doğrulama,
    hiperparametre ayarlama ve sonuç yorumlama adımlarını içerir.

Kütüphaneler:
    pandas, numpy, scikit-learn, matplotlib, seaborn

Çalıştırma adımları:
    1. pip install -r requirements.txt
    2. python telco_churn_ml_project.py
    CSV dosyası (WA_Fn-UseC_-Telco-Customer-Churn.csv) aynı klasörde olmalıdır.
"""

from __future__ import annotations

import csv
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Windows konsolunda Türkçe karakterlerin düzgün görünmesi için
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# -----------------------------------------------------------------------------
# Yardımcı: bozulmuş CSV satırlarını temizle
# -----------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RANDOM_STATE = 42


def _fix_european_number(token: str) -> str:
    """'2,985' / '""2,985""' gibi değerleri 29.85 formatına çevirir."""
    token = token.strip().strip('"')
    if re.fullmatch(r"\d+,\d+", token):
        digits = token.replace(",", "")
        # Ondalık ayırıcı yanlışlıkla binlik gibi yazılmış: son 2 hane kuruş
        return f"{digits[:-2]}.{digits[-2:]}" if len(digits) > 2 else f"0.{digits.zfill(2)}"
    return token


def load_telco_csv(path: Path) -> pd.DataFrame:
    """
    Veri setini okur. Bazı satırlar tırnak içine alınmış ve MonthlyCharges /
    TotalCharges alanlarında Avrupa tarzı sayı formatı kullanılmış; bunlar düzeltilir.
    """
    with open(path, encoding="utf-8") as f:
        header = next(csv.reader([f.readline().strip()]))
        records: list[list[str]] = []

        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Tüm satır tek bir alan gibi tırnaklanmışsa içeriği aç
            if line.startswith('"') and line.endswith('"'):
                inner = line[1:-1]
                # ""2,985"" -> sayısal token
                inner = re.sub(
                    r'""(\d+,\d+)""',
                    lambda m: _fix_european_number(m.group(1)),
                    inner,
                )
                inner = inner.replace('""', '"')
                parts = next(csv.reader([inner]))
            else:
                parts = next(csv.reader([line]))

            if len(parts) != len(header):
                continue

            # Hâlâ virgüllü kalan sayı alanlarını düzelt
            parts[18] = _fix_european_number(parts[18])  # MonthlyCharges
            parts[19] = _fix_european_number(parts[19])  # TotalCharges
            records.append(parts)

    df = pd.DataFrame(records, columns=header)
    return df


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # =========================================================================
    # 2–3. Veri okuma, problem tanımı, hedef değişken
    # =========================================================================
    section("2-3. Veri Seti ve Problem Tanımı")
    print(
        "Problem: Telekomünikasyon müşterilerinin churn (ayrılma) durumunu tahmin etmek.\n"
        "Problem türü: SINIFLANDIRMA (ikili — Yes/No).\n"
        "Hedef değişken: Churn"
    )

    df = load_telco_csv(DATA_PATH)
    print(f"\nVeri başarıyla okundu: {DATA_PATH.name}")

    # =========================================================================
    # 4. Temel veri inceleme
    # =========================================================================
    section("4. Temel Veri İnceleme")
    print("İlk 5 satır:")
    print(df.head())
    print(f"\nSatır-sütun sayısı: {df.shape[0]} satır, {df.shape[1]} sütun")
    print("\nVeri tipleri:")
    print(df.dtypes)
    print("\nTemel istatistikler (ham):")
    print(df.describe(include="all").T)

    # Sayısal dönüşüm
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    print("\nSayısal sütun istatistikleri:")
    print(df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].describe())
    print("\nHedef dağılımı:")
    print(df["Churn"].value_counts())
    print(df["Churn"].value_counts(normalize=True).round(3))

    # =========================================================================
    # 5. Eksik değer kontrolü ve temizleme
    # =========================================================================
    section("5. Eksik Değer Kontrolü")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() else "Eksik değer yok (NaN).")
    print(f"\nToplam eksik hücre: {int(df.isnull().sum().sum())}")

    # TotalCharges bazen boş olabilir (yeni müşteri); medyan ile doldur
    if df["TotalCharges"].isnull().any():
        median_tc = df["TotalCharges"].median()
        n_miss = int(df["TotalCharges"].isnull().sum())
        df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
        print(f"TotalCharges içindeki {n_miss} eksik değer medyan ({median_tc:.2f}) ile dolduruldu.")

    # Analizde kimlik sütunu kullanılmaz
    df = df.drop(columns=["customerID"])

    # Kalan satır eksiklerini düşür
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Kalıcı eksik satır temizliği: {before} -> {len(df)}")

    # =========================================================================
    # 7. Aykırı değer inceleme (IQR + winsorize / sınırlandırma)
    # =========================================================================
    section("7. Aykırı Değer İnceleme")
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(y=df[col], ax=ax, color="#4C78A8")
        ax.set_title(col)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "outlier_boxplots.png", dpi=120)
    plt.close(fig)
    print(f"Boxplot kaydedildi: {OUTPUT_DIR / 'outlier_boxplots.png'}")

    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        print(f"{col}: IQR dışı {n_out} gözlem | sınırlar=[{low:.2f}, {high:.2f}]")
        # Silmek yerine sınırlandırma (winsorize) — bilgi kaybını azaltır
        df[col] = df[col].clip(lower=low, upper=high)

    print("Aykırı değerler IQR sınırlarına kırpıldı (winsorize).")

    # =========================================================================
    # 9. Öznitelik mühendisliği (en az 2 anlamlı öznitelik)
    # =========================================================================
    section("9. Öznitelik Mühendisliği")

    # 1) Ortalama aylık harcama tahmini (toplam / tenure)
    df["AvgChargesPerMonth"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # 2) Ek hizmet sayısı (güvenlik, yedekleme, koruma, destek, TV, film)
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["NumExtraServices"] = df[service_cols].apply(
        lambda row: sum(v == "Yes" for v in row), axis=1
    )

    # 3) Kısa süreli müşteri bayrağı (tenure grubu)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-0.1, 12, 36, 72],
        labels=["0-12", "13-36", "37-72"],
    ).astype(str)

    # 4) Aylık ücret seviyesi
    df["ChargeLevel"] = pd.qcut(
        df["MonthlyCharges"], q=3, labels=["Low", "Mid", "High"]
    ).astype(str)

    print("Üretilen öznitelikler:")
    print(" - AvgChargesPerMonth: TotalCharges / tenure")
    print(" - NumExtraServices: aktif ek hizmet adedi")
    print(" - TenureGroup: müşteri süresi grubu")
    print(" - ChargeLevel: aylık ücret seviyesi (üç dilim)")
    print(df[["AvgChargesPerMonth", "NumExtraServices", "TenureGroup", "ChargeLevel"]].head())

    # =========================================================================
    # 6. Kategorik encoding
    # =========================================================================
    section("6. Kategorik Değişken Encoding")

    target = "Churn"
    y = (df[target] == "Yes").astype(int)
    X = df.drop(columns=[target])

    binary_map = {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    for col in binary_cols:
        X[col] = X[col].map(binary_map)

    # Çok sınıflı kategorikler: one-hot
    multi_cat = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
        "TenureGroup",
        "ChargeLevel",
    ]
    X = pd.get_dummies(X, columns=multi_cat, drop_first=True)
    # Tüm sütunları sayısal float'a çevir
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"Encoding sonrası öznitelik sayısı: {X.shape[1]}")
    print("Örnek sütunlar:", list(X.columns[:8]), "...")

    # =========================================================================
    # 10. Öznitelik seçimi
    # =========================================================================
    section("10. Öznitelik Seçimi")

    # a) Düşük varyanslı değişkenleri ele
    var_selector = VarianceThreshold(threshold=0.01)
    X_var = pd.DataFrame(
        var_selector.fit_transform(X),
        columns=X.columns[var_selector.get_support()],
        index=X.index,
    )
    print(f"VarianceThreshold sonrası: {X.shape[1]} -> {X_var.shape[1]}")

    # b) ANOVA F-test ile en iyi K öznitelik
    k = min(20, X_var.shape[1])
    kbest = SelectKBest(score_func=f_classif, k=k)
    X_selected = pd.DataFrame(
        kbest.fit_transform(X_var, y),
        columns=X_var.columns[kbest.get_support()],
        index=X_var.index,
    )
    scores = pd.Series(kbest.scores_, index=X_var.columns).sort_values(ascending=False)
    print(f"SelectKBest (k={k}) seçilen öznitelikler:")
    print(scores.head(k).round(2))

    # Korelasyon (seçilen özellikler arası)
    corr = X_selected.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Seçilen Öznitelikler Korelasyon Matrisi")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_correlation.png", dpi=120)
    plt.close(fig)

    feature_names = list(X_selected.columns)

    # =========================================================================
    # 11. Train / Validation / Test ayrımı (stratify)
    # =========================================================================
    section("11. Train / Validation / Test Ayrımı")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_selected, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )
    # Oranlar: %60 train, %20 val, %20 test
    print(f"Train: {X_train.shape[0]} | Validation: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print("Churn oranı (train/val/test):",
          f"{y_train.mean():.3f} / {y_val.mean():.3f} / {y_test.mean():.3f}")

    # =========================================================================
    # 8. Ölçekleme (KNN ve Logistic Regression için)
    # =========================================================================
    section("8. Ölçekleme (StandardScaler)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("Sayısal öznitelikler StandardScaler ile ölçeklendi (train'e fit).")

    # =========================================================================
    # 12–13. En az 3 model eğitimi ve validation karşılaştırması
    # =========================================================================
    section("12-13. Model Eğitimi ve Validation Karşılaştırması")

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            True,
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=5),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, max_depth=8
            ),
            False,
        ),
    }

    val_results = []
    fitted = {}

    for name, (model, use_scaled) in models.items():
        Xtr = X_train_scaled if use_scaled else X_train
        Xva = X_val_scaled if use_scaled else X_val
        model.fit(Xtr, y_train)
        pred = model.predict(Xva)
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_val, pred),
            "Precision": precision_score(y_val, pred, zero_division=0),
            "Recall": recall_score(y_val, pred, zero_division=0),
            "F1": f1_score(y_val, pred, zero_division=0),
        }
        val_results.append(metrics)
        fitted[name] = (model, use_scaled)
        print(
            f"{name:22s} | Acc={metrics['Accuracy']:.3f} | "
            f"Prec={metrics['Precision']:.3f} | Rec={metrics['Recall']:.3f} | "
            f"F1={metrics['F1']:.3f}"
        )

    results_df = pd.DataFrame(val_results).sort_values("F1", ascending=False)
    print("\nValidation sıralaması (F1'e göre):")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    print(f"\nValidation'a göre en iyi model: {best_name}")

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]]
    plot_df.plot(kind="bar", ax=ax, rot=15)
    ax.set_ylim(0, 1)
    ax.set_title("Validation Metrik Karşılaştırması")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "validation_comparison.png", dpi=120)
    plt.close(fig)

    # =========================================================================
    # 14. Hiperparametre ayarlama (Grid Search) — en iyi model ailesi
    # =========================================================================
    section("14. Hiperparametre Ayarlama (GridSearchCV)")

    # En iyi modele göre arama uzayı; genelde RF veya LR seçilir
    if best_name == "Random Forest":
        base = RandomForestClassifier(random_state=RANDOM_STATE)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8, None],
            "min_samples_split": [2, 5],
        }
        search_X = X_temp  # train+val
        search_y = y_temp
        use_scaled_best = False
    elif best_name == "Logistic Regression":
        base = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        }
        search_X = scaler.fit_transform(X_temp)
        search_y = y_temp
        use_scaled_best = True
    elif best_name == "KNN":
        base = KNeighborsClassifier()
        param_grid = {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }
        search_X = scaler.fit_transform(X_temp)
        search_y = y_temp
        use_scaled_best = True
    else:  # Decision Tree
        base = DecisionTreeClassifier(random_state=RANDOM_STATE)
        param_grid = {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        }
        search_X = X_temp
        search_y = y_temp
        use_scaled_best = False

    grid = GridSearchCV(
        base,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(search_X, search_y)
    print(f"En iyi parametreler: {grid.best_params_}")
    print(f"CV F1 (en iyi): {grid.best_score_:.4f}")
    best_model = grid.best_estimator_

    # Scaler'ı tüm train+val üzerinde yeniden fit et (test için)
    scaler_final = StandardScaler()
    X_temp_scaled = scaler_final.fit_transform(X_temp)
    X_test_scaled_final = scaler_final.transform(X_test)

    if use_scaled_best:
        best_model.fit(X_temp_scaled, y_temp)
        y_pred = best_model.predict(X_test_scaled_final)
        explain_X = X_temp_scaled
    else:
        best_model.fit(X_temp, y_temp)
        y_pred = best_model.predict(X_test)
        explain_X = X_temp

    # =========================================================================
    # 15. Test değerlendirmesi
    # =========================================================================
    section("15. Test Seti Değerlendirmesi")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Model: {best_name} (Grid Search sonrası)")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # =========================================================================
    # 17. Bonus: Açıklanabilirlik (feature importance / katsayı)
    # =========================================================================
    section("17. Bonus — Model Açıklanabilirliği")

    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=feature_names)
        imp = imp.sort_values(ascending=False)
        print("Feature Importance (top 10):")
        print(imp.head(10).round(4))
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.head(10).sort_values().plot(kind="barh", ax=ax, color="#4C78A8")
        ax.set_title("En Önemli 10 Öznitelik")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=120)
        plt.close(fig)
        top_features = list(imp.head(5).index)
    elif hasattr(best_model, "coef_"):
        coef = pd.Series(best_model.coef_.ravel(), index=feature_names)
        coef_abs = coef.reindex(coef.abs().sort_values(ascending=False).index)
        print("Logistic Regression katsayıları (mutlak değere göre top 10):")
        print(coef_abs.head(10).round(4))
        fig, ax = plt.subplots(figsize=(8, 5))
        coef_abs.head(10).sort_values().plot(kind="barh", ax=ax, color="#4C78A8")
        ax.set_title("En Etkili 10 Katsayı")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=120)
        plt.close(fig)
        top_features = list(coef_abs.head(5).index)
    else:
        # KNN için proxy: SelectKBest skorları
        print("KNN için doğrudan importance yok; SelectKBest skorları kullanılıyor.")
        print(scores.head(10).round(2))
        top_features = list(scores.head(5).index)

    # =========================================================================
    # 16. Sonuç yorumu
    # =========================================================================
    section("16. Model Sonucu Yorumu")
    print(
        f"""
Özet:
- En iyi model (validation F1): {best_name}
- Grid Search sonrası test Accuracy={acc:.3f}, Precision={prec:.3f},
  Recall={rec:.3f}, F1={f1:.3f}
- Öne çıkan değişkenler: {', '.join(top_features)}
- Contract tipi, tenure/süre grubu, internet hizmeti ve ek hizmetler
  churn ile ilişkili görünüyor; kısa süreli ve aylık sözleşmeli müşteriler
  genelde daha yüksek ayrılma riski taşır.

Sınırlılıklar:
- Veri seti ~500 satır; genelleme gücü sınırlı olabilir.
- Sınıf dengesizliği (churn oranı ~%25) recall/precision dengesini etkiler.
- Bazı CSV satırlarında sayı formatı bozulmuştu; düzeltilmiş olsa da
  kaynak veri kalitesi modele gürültü ekleyebilir.
- Harici ekonomik/rekabet faktörleri modelde yok.
"""
    )
    print(f"Grafikler '{OUTPUT_DIR}' klasörüne kaydedildi.")
    print("Proje başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
