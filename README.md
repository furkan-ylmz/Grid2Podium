<div align="center">

[English](#english) | [Türkçe](#türkçe)

</div>

---

<a name="english"></a>
# Formula 1 Race Outcome Prediction System

Grid2Podium is an end-to-end deep learning framework designed to predict Formula 1 race finishing tiers (**Podium**, **Points**, or **No Points / DNF**) using tabular race data spanning the 2019 to 2026 seasons.

By analyzing multi-dimensional race factors—such as starting grid positions, circuit characteristics, driver skills, constructor performance, and seasonal dynamics—the system trains, benchmarks, and serves 5 distinct neural network architectures via an interactive Streamlit web dashboard.

---

## Data Pipeline & Preprocessing Methodology

The data pipeline standardizes and prepares raw historical race logs for tabular neural network training:

1. **Ingestion & Season Parsing:** Dynamic extraction of seasonal metadata from raw CSV logs spanning 2019 through 2026.
2. **Target Categorization:** Multi-class target assignment based on official finishing classification:
   - **Tier 0 (Podium):** Positions 1 to 3
   - **Tier 1 (Points):** Positions 4 to 10
   - **Tier 2 (No Points / DNF):** Positions 11+ and retirements
3. **Feature Cleaning:** Non-predictive in-race features (`Laps`, `Points`, `Fastest Lap Time`, etc.) are eliminated to prevent data leakage. Starting grid values are normalized and cleaned.
4. **Encoding & Stratification:**
   - Categorical entities (`Track`, `Driver`, `Team`) are mapped using `LabelEncoder` and subsequently one-hot encoded for neural input.
   - Stratified dataset partitioning by season (`Year`) ensures uniform class and season distribution:
     - **Training Set (70%)**
     - **Validation Set (20%)**
     - **Test Benchmark Set (10%)**

---

## Deep Learning Architectures

Five specialized deep learning architectures are benchmarked on the tabular race dataset:

1. **Custom MLP (Multi-Layer Perceptron):**
   - 4-layer fully connected architecture with Batch Normalization, ReLU activations, and progressive Dropout (0.5 to 0.3) for regularized feature representation.
2. **Simple LSTM (PyTorch Built-in):**
   - Sequential recurrent network leveraging `nn.LSTM` to model tabular feature interdependencies.
3. **Manual Gate LSTM:**
   - Custom low-level LSTM engineered from mathematical foundations, explicitly computing input ($i_t$), forget ($f_t$), candidate cell ($g_t$), and output ($o_t$) gate activations without PyTorch cell abstractions.
4. **1D CNN (Convolutional Neural Network):**
   - Multi-stage 1D convolutional layers with batch normalization and max-pooling designed to capture localized feature interactions across tabular dimensions.
5. **FT-Transformer (Tabular Transformer):**
   - Multi-head self-attention architecture tailored for tabular tokens, evaluating attention-driven feature interactions.

---

## Model Performance & Evaluation

All models are trained with Cross-Entropy Loss, Adam optimization, and `ReduceLROnPlateau` dynamic learning rate scheduling paired with Early Stopping.

### 1. Comparative Evaluation Metrics
Comprehensive performance breakdown across Training, Validation, and Test sets:

![Model Evaluation Metrics](results/model_evaluation_metrics.png)

### 2. Loss & Accuracy Learning Dynamics
**Loss Curves:**
![Loss Curves](results/loss_curves.png)

**Accuracy Curves:**
![Accuracy Curves](results/accuracy_curves.png)

### 3. Test Confusion Matrices
Visualizing true positive rates and classification error distributions across all 3 tiers:

![Test Confusion Matrices](results/test_confusion_matrices.png)

---

## Project Structure

```
Grid2Podium/
├── data/                                 # Central data directory
│   ├── raw/                              # Standardized raw CSV season results (2019-2026)
│   │   ├── formula1_2019_season_race_results.csv
│   │   ├── formula1_2020_season_race_results.csv
│   │   ├── ...
│   │   └── formula1_2026_season_race_results.csv
│   └── processed/                        # Preprocessed and split datasets
│       ├── encoders.pkl                  # Fitted LabelEncoder objects
│       ├── train.csv                     # Training set (70%)
│       ├── val.csv                       # Validation set (20%)
│       └── test.csv                      # Test set (10%)
├── core/                                 # Core modules
│   ├── models.py                         # Neural model definitions (MLP, LSTM, Manual LSTM, CNN1D, Transformer)
│   ├── preprocessing.py                  # Ingestion, cleaning, encoding, and DataLoader pipeline
│   ├── trainer.py                        # Model training loop, validation, and Early Stopping
│   ├── evaluator.py                      # Multi-metric evaluation and confusion matrix engine
│   ├── visualization.py                  # Learning curves, confusion heatmaps, and metric table generator
│   ├── predictor.py                      # Live inference engine (RacePredictor)
│   └── train.py                          # Multi-model training and evaluation orchestrator
├── web/                                  # Web presentation layer
│   └── app.py                            # Streamlit interactive prediction dashboard
├── models/                               # Serialized model artifacts
│   ├── best_model.pth                    # State dictionary of the top-performing model
│   ├── best_model_arch.pkl               # Name of the best model architecture
│   └── feature_columns.pkl               # One-hot encoded feature schema
├── results/                              # Generated visual artifacts
│   ├── accuracy_curves.png
│   ├── loss_curves.png
│   ├── model_evaluation_metrics.png
│   ├── test_confusion_matrices.png
│   └── validation_confusion_matrices.png
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Git ignore rules
└── README.md                             # Documentation
```

---

## Installation & Usage

### 1. Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preprocessing
Process raw CSV logs, generate encodings, and split datasets:
```bash
python core/preprocessing.py
```

### 3. Model Training & Evaluation
Train all 5 architectures, benchmark results, save the best-performing model, and generate analytical charts:
```bash
python core/train.py
```

### 4. Interactive Web Application
Launch the Streamlit prediction interface:
```bash
streamlit run web/app.py
```

---

## Dataset & Attribution
Raw historical race data is obtained from [Formula 1 Datasets (GitHub)](https://github.com/toUpperCase78/formula1-datasets).

<br>

---

<a name="türkçe"></a>
# Formula 1 Yarış Sonucu Tahmin Sistemi

Grid2Podium, 2019-2026 Formula 1 sezonlarına ait yarış verilerini kullanarak pilotların yarış sonu başarılarını (**Podyum**, **Puan** veya **Puansız / DNF**) tahmin eden uçtan uca bir derin öğrenme sistemidir.

Başlangıç pozisyonu (grid), pist özellikleri, sürücü, takım performansı ve sezon dinamikleri gibi çok boyutlu parametreleri analiz eden sistem; 5 farklı yapay sinir ağı mimarisini karşılaştırmalı olarak eğitir ve en başarılı modeli etkileşimli bir Streamlit web paneli üzerinden sunar.

---

## Veri Hattı ve Ön İşleme Metodolojisi

Veri hattı, ham yarış verilerini tablosal derin öğrenme modellerine uygun hale getirmek için şu adımları uygular:

1. **Veri Toplama ve Sezon Ayrıştırma:** 2019'dan 2026'ya kadar olan ham CSV dosyaları okunur ve sezon bilgisi dinamik olarak çıkartılır.
2. **Hedef Sınıflandırma (Target Mapping):** Yarış bitiş pozisyonları 3 ana başarı seviyesine ayrılır:
   - **Seviye 0 (Podyum):** 1. ile 3. sıra arası
   - **Seviye 1 (Puan):** 4. ile 10. sıra arası
   - **Seviye 2 (Puansız / DNF):** 11. sıra ve sonrası ile yarışı tamamlayamayanlar
3. **Veri Temizleme:** Bilgi sızıntısını (data leakage) önlemek amacıyla yarış bittikten sonra oluşan sütunlar (`Laps`, `Points`, `Fastest Lap Time` vb.) çıkartılır. Başlangıç grid pozisyonları normalize edilir.
4. **Kodlama ve Katmanlı Bölme:**
   - Kategorik değişkenler (`Track`, `Driver`, `Team`) `LabelEncoder` ile sayısallaştırılır ve ardından One-Hot Encoding ile modele hazır hale getirilir.
   - Sınıf ve sezon dağılımını korumak için katmanlı bölme (Stratified Split by Year) uygulanır:
     - **Eğitim Seti (%70)**
     - **Doğrulama Seti (%20)**
     - **Test Seti (%10)**

---

## Kullanılan Derin Öğrenme Mimarileri

Tablosal yarış verileri üzerinde 5 farklı sinir ağı mimarisi eğitilerek performansları kıyaslanmaktadır:

1. **Custom MLP (Çok Katmanlı Algılayıcı):**
   - Tablosal özellikleri öğrenmek için Batch Normalization, ReLU aktivasyonları ve kademeli Dropout (%50-%30) içeren 4 katmanlı fully connected ağ.
2. **Hazır LSTM (PyTorch Built-in):**
   - Tablosal özellik uzayındaki sıralı ve zamansal bağımlılıkları modelleyen `nn.LSTM` tabanlı ağ.
3. **Manuel LSTM:**
   - PyTorch hazır fonksiyonları kullanılmadan; giriş ($i_t$), unutma ($f_t$), aday hücre ($g_t$) ve çıkış ($o_t$) kapı tensörleri matematiksel formülleriyle sıfırdan açık olarak kodlanan LSTM mimarisi.
4. **1D CNN (1 Boyutlu Evrişimli Ağ):**
   - Özellik vektörleri üzerinde yerel ilişkileri çıkaran çok aşamalı tek boyutlu evrişim ve havuzlama (max-pooling) katmanları.
5. **FT-Transformer (Tablosal Transformer):**
   - Tablosal tokenler üzerinde çok başlı öz-dikkat (Multi-head Self-Attention) mekanizmasının başarımını test eden Transformer mimarisi.

---

## Model Performans Analizi ve Sonuçlar

Tüm modeller Cross-Entropy Loss, Adam optimizasyonu ve Early Stopping ile desteklenen `ReduceLROnPlateau` öğrenme oranı planlayıcısı ile eğitilmiştir.

### 1. Karşılaştırmalı Model Değerlendirme Tablosu
Modellerin Eğitim, Doğrulama ve Test setlerindeki ayrıntılı başarım metrikleri:

![Model Performans Metrikleri](results/model_evaluation_metrics.png)

### 2. Öğrenme Dinamikleri (Kayıp ve Doğruluk Grafikleri)
**Kayıp (Loss) Eğrileri:**
![Kayıp Grafikleri](results/loss_curves.png)

**Doğruluk (Accuracy) Eğrileri:**
![Doğruluk Grafikleri](results/accuracy_curves.png)

### 3. Test Seti Konfüzyon Matrisleri
Hedef sınıflar üzerindeki doğru tahmin oranları ve hata dağılımları:

![Test Konfüzyon Matrisleri](results/test_confusion_matrices.png)

---

## Proje Dizin Yapısı

```
Grid2Podium/
├── data/                                 # Veri depolama dizini
│   ├── raw/                              # Standart formatta ham sezon verileri (2019-2026)
│   │   ├── formula1_2019_season_race_results.csv
│   │   ├── formula1_2020_season_race_results.csv
│   │   ├── ...
│   │   └── formula1_2026_season_race_results.csv
│   └── processed/                        # İşlenmiş ve ayrıştırılmış veri setleri
│       ├── encoders.pkl                  # LabelEncoder nesneleri
│       ├── train.csv                     # Eğitim seti (%70)
│       ├── val.csv                       # Doğrulama seti (%20)
│       └── test.csv                      # Test seti (%10)
├── core/                                 # Çekirdek yapay zeka ve işlem modülleri
│   ├── models.py                         # PyTorch model tanımları (MLP, LSTM, Manual LSTM, CNN1D, Transformer)
│   ├── preprocessing.py                  # Veri okuma, temizleme, One-Hot dönüşümü ve DataLoader üretimi
│   ├── trainer.py                        # Model eğitim döngüsü, doğrulama ve Early Stopping
│   ├── evaluator.py                      # Metrik hesaplama ve karmaşıklık matrisi motoru
│   ├── visualization.py                  # Kayıp, doğruluk grafikleri, ısı haritaları ve metrik tablosu
│   ├── predictor.py                      # Canlı tahmin motoru (RacePredictor)
│   └── train.py                          # Toplu model eğitim ve değerlendirme yöneticisi
├── web/                                  # Web sunum katmanı
│   └── app.py                            # Streamlit web tahmin paneli
├── models/                               # Kaydedilen model ağırlıkları ve metadata
│   ├── best_model.pth                    # En yüksek test doğruluğuna sahip model ağırlıkları
│   ├── best_model_arch.pkl               # En iyi model mimari adı
│   └── feature_columns.pkl               # One-hot kodlanmış özellik listesi
├── results/                              # Grafik çıktıları ve metrik tablosu
│   ├── accuracy_curves.png
│   ├── loss_curves.png
│   ├── model_evaluation_metrics.png
│   ├── test_confusion_matrices.png
│   └── validation_confusion_matrices.png
├── requirements.txt                      # Bağımlılık listesi
├── .gitignore                            # Git yok sayma kuralları
└── README.md                             # Dokümantasyon
```

---

## Kurulum ve Çalıştırma

### 1. Bağımlılıkların Yüklenmesi
Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

### 2. Veri Ön İşleme (Data Preprocessing)
Ham verileri işleyip eğitim, doğrulama ve test setlerini oluşturun:
```bash
python core/preprocessing.py
```

### 3. Model Eğitimi ve Değerlendirme (Training)
5 derin öğrenme mimarisini eğitin, sonuçları karşılaştırın ve en iyi modeli kaydedin:
```bash
python core/train.py
```

### 4. Web Tahmin Uygulaması (Streamlit)
Etkileşimli web arayüzünü başlatın:
```bash
streamlit run web/app.py
```

---

## Veri Seti ve Referanslar
Ham yarış sonuçları verisi [Formula 1 Datasets (GitHub)](https://github.com/toUpperCase78/formula1-datasets) kaynağından temin edilmiştir.
