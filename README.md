# Telco Müşteri Ayrılma (Churn) Tahmini

**Türkiye Yapay Zeka Akademisi — Makine Öğrenmesi Final Ödevi**  
**Ad Soyad:** Yunus Büyükkafes

Uçtan uca makine öğrenmesi sınıflandırma projesi.

---

## Amaç

Telekomünikasyon müşterilerinin hizmeti bırakıp bırakmayacağını (`Churn`) tahmin eden bir sınıflandırma modeli geliştirmek. Proje; veri inceleme, ön işleme, öznitelik mühendisliği, model eğitimi, validation karşılaştırması, çapraz doğrulama, hiperparametre ayarlama ve test değerlendirmesini kapsar.

---

## Veri Seti

| Özellik | Açıklama |
|---------|----------|
| **Dosya** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Kaynak** | IBM Telco Customer Churn |
| **Boyut** | ~500 satır, 21 sütun |
| **Problem türü** | İkili sınıflandırma (`Churn`: Yes / No) |
| **Hedef değişken** | `Churn` |

**Problem:** Müşterinin telekom hizmetini bırakıp bırakmayacağını tahmin etmek.

> CSV'de bazı satırlarda Avrupa tarzı sayı formatı ve tırnak sarmalama vardır; kod bunları otomatik düzeltir.

---

## Repository Yapısı

```
├── telco_churn_ml_project.py   # Ana Python script (yerel çalıştırma)
├── Telco_Churn_Odev.ipynb      # Google Colab notebook (alternatif)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

---

## Nasıl Çalıştırılır

### Yerel ortam (Python script)

```bash
pip install -r requirements.txt
python telco_churn_ml_project.py
```

Grafikler `outputs/` klasörüne kaydedilir.

### Google Colab (notebook)

1. [Google Colab](https://colab.research.google.com/) açın
2. `Telco_Churn_Odev.ipynb` dosyasını yükleyin
3. **Runtime → Run all**
4. CSV yükleme adımında `WA_Fn-UseC_-Telco-Customer-Churn.csv` dosyasını seçin

---

## Ödev Maddeleri Karşılığı

| # | Madde | Uygulama |
|---|-------|----------|
| 1 | Docstring | `telco_churn_ml_project.py` başında |
| 2 | Veri okuma | `load_telco_csv()` ile pandas |
| 3 | Hedef değişken | `Churn` — ikili sınıflandırma |
| 4 | EDA | head, shape, dtypes, describe |
| 5 | Eksik değer | TotalCharges medyan doldurma, dropna |
| 6 | Encoding | Binary map + one-hot (`get_dummies`) |
| 7 | Aykırı değer | IQR analizi + winsorize (clip) |
| 8 | Ölçekleme | StandardScaler (LR, KNN için) |
| 9 | Feature engineering | AvgChargesPerMonth, NumExtraServices, TenureGroup, ChargeLevel |
| 10 | Öznitelik seçimi | VarianceThreshold + SelectKBest |
| 11 | Veri ayrımı | Train %60 / Val %20 / Test %20, stratify |
| 12 | 3+ model | Logistic Regression, KNN, Decision Tree, Random Forest |
| 13 | Validation karşılaştırma | Accuracy, Precision, Recall, F1 |
| 14 | Hiperparametre | GridSearchCV (cv=5, scoring=f1) |
| 15 | Test değerlendirme | Confusion matrix, accuracy, precision, recall, F1 |
| 16 | Sonuç yorumu | Script/notebook sonunda yazdırılır |
| 17 | Bonus | Feature importance / katsayı analizi |

---

## Sonuç Özeti

**Validation'a göre en iyi model:** Logistic Regression

**Grid Search sonrası test performansı:**

| Metrik | Değer |
|--------|-------|
| Accuracy | 0.76 |
| Precision | 0.52 |
| Recall | 0.48 |
| F1-score | 0.50 |

**Öne çıkan değişkenler:** InternetService_Fiber optic, tenure, Contract_Two year, TechSupport_Yes, Contract_One year

**Yorum:** Sözleşme tipi, müşteri süresi (tenure) ve internet hizmeti churn tahmininde en etkili faktörlerdir. Kısa süreli ve aylık sözleşmeli müşteriler daha yüksek ayrılma riski taşır.

**Sınırlılıklar:**
- Veri seti ~500 satır; genelleme gücü sınırlı olabilir
- Sınıf dengesizliği (churn ~%25) precision/recall dengesini etkiler
- Harici ekonomik/rekabet faktörleri modelde yer almaz

---

## Kullanılan Kütüphaneler

pandas, numpy, scikit-learn, matplotlib, seaborn

---
