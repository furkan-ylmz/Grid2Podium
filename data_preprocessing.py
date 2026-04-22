import pandas as pd
import glob
import os
import re
from sklearn.preprocessing import LabelEncoder
import numpy as np

def extract_year_from_filename(filename):
    # Dosya adından yılı çek (ör: formula1_2019season_raceResults.csv -> 2019)
    match = re.search(r'(\d{4})[sS]eason', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    print("Veriler yükleniyor...")
    file_pattern = os.path.join('datasets', '*.csv')
    file_list = glob.glob(file_pattern)
    
    df_list = []
    for file in file_list:
        year = extract_year_from_filename(file)
        if year is None:
            continue
        df = pd.read_csv(file)
        df['Year'] = year
        df_list.append(df)
        
    if not df_list:
        print("Hata: CSV dosyası bulunamadı!")
        return
        
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Toplam {len(full_df)} satır veri yüklendi.")
    
    # 1. Hedef Sütunu (Position) Temizleme ve Sınıflama (0: Podyum, 1: Puan, 2: Puansız/DNF)
    def categorize_position(pos):
        try:
            pos_int = int(pos)
            if pos_int <= 3:
                return 0
            elif pos_int <= 10:
                return 1
            else:
                return 2
        except ValueError:
            # NC (Not Classified), DQ (Disqualified), Ret (Retired) vb. bitiremeyenler
            return 2
            
    full_df['Target_Tier'] = full_df['Position'].apply(categorize_position)
    
    # 2. Gereksiz ve Veri Sızıntısına (Leakage) Yol Açan Sütunları Çıkarma
    # 'No' (Araç numarası) da genelde tahmin için anlamsızdır, onu da çıkarabiliriz.
    columns_to_drop = ['Laps', 'Total Time/Gap/Retirement', 'Points', 'Fastest Lap', 'Position', 'No', 'Time/Retired', '+1 Pt', 'Set Fastest Lap', 'Fastest Lap Time']
    # Dosyalarda sütun isimleri biraz farklı olabilir, sadece var olanları çıkar.
    cols_to_drop_existing = [col for col in columns_to_drop if col in full_df.columns]
    full_df.drop(columns=cols_to_drop_existing, inplace=True)
    print(f"Çıkartılan sütunlar: {cols_to_drop_existing}")
    
    # 3. Starting Grid (Başlangıç Pozisyonu) Temizleme
    # Bazen "Pit Lane" veya eksik veri olabilir. Bunları en sonuncu sıradan başlatıyoruz gibi varsayabiliriz (örn: 20)
    def clean_grid(grid):
        try:
            return int(grid)
        except ValueError:
            return 20 # Pit lane start veya benzeri
            
    full_df['Starting Grid'] = full_df['Starting Grid'].apply(clean_grid)
    
    # 4. Kategorik Değişkenleri Encode Etme
    categorical_cols = ['Track', 'Driver', 'Team']
    encoders = {}
    
    for col in categorical_cols:
        full_df[col] = full_df[col].astype(str)
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col])
        encoders[col] = le
        
    print("Kategorik değişkenler (Track, Driver, Team) sayısal değerlere dönüştürüldü.")
    
    # 5. Veriyi Train, Validation ve Test Olarak Bölme (%70 Train, %20 Val, %10 Test)
    from sklearn.model_selection import train_test_split
    
    # İlk olarak veriyi Train (%70) ve Kalanlar (%30) olarak ayırıyoruz. Yıllara göre stratify (tabakalı) yapıyoruz.
    train_df, temp_df = train_test_split(full_df, test_size=0.30, stratify=full_df['Year'], random_state=42)
    
    # Kalan %30'luk kısmı Validation (2/3 -> %20) ve Test (1/3 -> %10) olarak ayırıyoruz.
    val_df, test_df = train_test_split(temp_df, test_size=(1/3), stratify=temp_df['Year'], random_state=42)
    
    # Klasör yoksa oluştur
    os.makedirs('processed_data', exist_ok=True)
    
    # Encoder'ları web app için kaydet
    import pickle
    with open('processed_data/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
        
    # Verileri kaydet
    train_df.to_csv('processed_data/train.csv', index=False)
    val_df.to_csv('processed_data/val.csv', index=False)
    test_df.to_csv('processed_data/test.csv', index=False)
    
    print("\nVeri Hazırlığı Tamamlandı!")
    print(f"Train Seti (%70): {len(train_df)} satır")
    print(f"Validation Seti (%20): {len(val_df)} satır")
    print(f"Test Seti (%10): {len(test_df)} satır")
    print(f"\nKullanılan Özellikler (Features): {list(train_df.columns)}")

if __name__ == "__main__":
    main()
