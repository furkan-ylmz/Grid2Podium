# BIM 430 Derin Ogrenme Proje Odevi: Formula 1 Yaris Sonucu Tahmin Sistemi

## Proje Hakkinda
Bu proje, 2019-2026 yillari arasindaki Formula 1 yaris verilerini kullanarak, yaris sonuclari uzerine bir siniflandirma tahmini yapmayi amaclamaktadir. Proje kapsaminda uc farkli derin ogrenme modeli gelistirilmis ve performanslari karsilastirilmistir.

## Problem Tanimi
Yaris sonuclari, veri setindeki dengesizligi azaltmak ve model basarisini artirmak amaciyla uc ana kategoriye (Tier) ayrilmistir:
- Sinif 0 (Podyum): 1., 2. ve 3. sirada bitirenler.
- Sinif 1 (Puan Alanlar): 4. ile 10. sira arasinda bitirenler.
- Sinif 2 (Puansiz/Bitiremeyenler): 11. sira ve ustu veya yarisi tamamlayamayanlar.

## Kullanilan Teknolojiler
- Dil: Python 3.11+
- Derin Ogrenme Kutuphanesi: PyTorch
- Veri Isleme: Pandas, NumPy, Scikit-learn
- Gorsellestirme: Matplotlib, Seaborn
- Arayuz: Streamlit

## Proje Yapisi
- data_preprocessing.py: Veri setlerinin birlestirilmesi, veri sizintisi (leakage) yapan sutunlarin temizlenmesi ve verinin %70 egitim, %20 dogrulama, %10 test olarak bolunmesi islemlerini yapar.
- train_models.py: Uc farkli modelin (Ozel MLP, LSTM, Wide & Deep) egitilmesi, test edilmesi ve karsilastirmali grafiklerin olusturulmasi sureclerini yonetir.
- app.py: Egitilmis en basarili modelin (LSTM) kullanici tarafindan test edilebilmesini saglayan web arayuzu dosyasidir.
- datasets/: Ham CSV dosyalarinin bulundugu dizin.
- processed_data/: Islenmis verilerin ve encoder dosyalarinin saklandigi dizin.
- models/: Egitilmis model agirliklarinin ve ozellik sutunlarinin saklandigi dizin.
- results/: Egitim kaybi, dogruluk ve confusion matrix grafiklerinin kaydedildigi dizin.

## Model Mimarileri
1. Ozel MLP (Ogrenci Tasarimi): BatchNorm, Dropout ve ReLU aktivasyon fonksiyonlari ile desteklenmis cok katmanli algilayici mimarisi.
2. LSTM: F1 verilerindeki sirali yapiyi ve takvimsel etkileri yakalamak amaciyla kullanilan uzun kisa sureli bellek agi.
3. Wide & Deep: Tablosal verilerde hem genis ozellikleri hem de derin ogrenme modellerinin yakaladigi karmasik iliskileri birlestiren mimari.

## Kurulum ve Calistirma
Sistemi yerelde calistirmak icin asagidaki adimlari izleyin:

1. Gerekli kutuphaneleri yukleyin:
   pip install torch pandas numpy scikit-learn matplotlib seaborn streamlit

2. Veri on isleme adimini calistirin:
   python data_preprocessing.py

3. Modelleri egitin:
   python train_models.py

4. Web arayuzunu baslatin:
   streamlit run app.py

## Sonuclar
Yapilan deneyler sonucunda LSTM modeli, yarislarin sirali yapisini ve surucu/takim devamligini capture etme yetenegi sayesinde yaklasik %74 dogruluk orani ile en yuksek performansi gostermistir. Detayli karsilastirmalar ve metrikler (Accuracy, Precision, Recall, F1-Score) 'results' klasoru altindaki grafiklerde sunulmustur.
