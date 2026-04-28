# Formula 1 Yarış Sonucu Tahmin Sistemi

## Proje Hakkında
Bu proje, 2019-2026 yılları arasındaki Formula 1 yarış verilerini kullanarak, yarış sonuçları üzerine bir sınıflandırma tahmini yapmayı amaçlamaktadır. Proje kapsamında üç farklı derin öğrenme modeli geliştirilmiş ve performansları karşılaştırılmıştır.

## Problem Tanımı
Yarış sonuçları, veri setindeki dengesizliği azaltmak ve model başarısını artırmak amacıyla üç ana kategoriye (Tier) ayrılmıştır:
- Sınıf 0 (Podyum): 1., 2. ve 3. sırada bitirenler.
- Sınıf 1 (Puan Alanlar): 4. ile 10. sıra arasında bitirenler.
- Sınıf 2 (Puansız/Bitiremeyenler): 11. sıra ve üstü veya yarışı tamamlayamayanlar.

## Kullanılan Teknolojiler
- Dil: Python 3.11+
- Derin Öğrenme Kütüphanesi: PyTorch
- Veri İşleme: Pandas, NumPy, Scikit-learn
- Görselleştirme: Matplotlib, Seaborn
- Arayüz: Streamlit

## Proje Yapısı
- data_preprocessing.py: Veri setlerinin birleştirilmesi, veri sızıntısı (leakage) yapan sütunların temizlenmesi ve verinin %70 eğitim, %20 doğrulama, %10 test olarak bölünmesi işlemlerini yapar.
- train_models.py: Üç farklı modelin (MLP, LSTM, Wide & Deep) eğitilmesi, test edilmesi ve karşılaştırmalı grafiklerin oluşturulması süreçlerini yönetir.
- app.py: Eğitilmiş en başarılı modelin kullanıcı tarafından test edilebilmesini sağlayan web arayüzü dosyasıdır.
- datasets/: Ham CSV dosyalarının bulunduğu dizin.
- processed_data/: İşlenmiş verilerin ve encoder dosyalarının saklandığı dizin.
- models/: Eğitilmiş model ağırlıklarının ve özellik sütunlarının saklandığı dizin.
- results/: Eğitim kaybı, doğruluk ve confusion matrix grafiklerinin kaydedildiği dizin.

## Model Mimarileri
1. MLP: BatchNorm, Dropout ve ReLU aktivasyon fonksiyonları ile desteklenmiş çok katmanlı algılayıcı mimarisi.
2. LSTM: F1 verilerindeki sıralı yapıyı ve takvimsel etkileri yakalamak amacıyla kullanılan uzun kısa süreli bellek ağı.
3. Wide & Deep: Tablosal verilerde hem geniş özellikleri hem de derin öğrenme modellerinin yakaladığı karmaşık ilişkileri birleştiren mimari.

## Kurulum ve Çalıştırma
Sistemi yerelde çalıştırmak için aşağıdaki adımları izleyin:

1. Gerekli kütüphaneleri yükleyin:
   pip install torch pandas numpy scikit-learn matplotlib seaborn streamlit

2. Veri ön işleme adımını çalıştırın:
   python data_preprocessing.py

3. Modelleri eğitin:
   python train_models.py

4. Web arayüzünü başlatın:
   streamlit run app.py

## Sonuçlar
Projenin sonucunda eğitilmiş yapay zeka modelleri ile yarışçıların podyum ve puan tahminleri başarılı bir şekilde sunulmaktadır. Çeşitli parametreler modifiye edilerek başarı oranı (Accuracy) grafiklerle ve metriklerle (Accuracy, Precision, Recall, F1-Score) 'results' klasörü altına kaydedilmektedir.
