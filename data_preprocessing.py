import pandas as pd
import glob
import os
import re
from sklearn.preprocessing import LabelEncoder
import numpy as np

def extract_year_from_filename(filename):
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
            return 2
            
    full_df['Target_Tier'] = full_df['Position'].apply(categorize_position)
    
    columns_to_drop = ['Laps', 'Total Time/Gap/Retirement', 'Points', 'Fastest Lap', 'Position', 'No', 'Time/Retired', '+1 Pt', 'Set Fastest Lap', 'Fastest Lap Time']
    cols_to_drop_existing = [col for col in columns_to_drop if col in full_df.columns]
    full_df.drop(columns=cols_to_drop_existing, inplace=True)
    print(f"Çıkartılan sütunlar: {cols_to_drop_existing}")
    
    def clean_grid(grid):
        try:
            return int(grid)
        except ValueError:
            return 20
            
    full_df['Starting Grid'] = full_df['Starting Grid'].apply(clean_grid)

    categorical_cols = ['Track', 'Driver', 'Team']
    encoders = {}
    
    for col in categorical_cols:
        full_df[col] = full_df[col].astype(str)
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col])
        encoders[col] = le
        
    print("Kategorik değişkenler (Track, Driver, Team) sayısal değerlere dönüştürüldü.")
    
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(full_df, test_size=0.30, stratify=full_df['Year'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=(1/3), stratify=temp_df['Year'], random_state=42)
    
    os.makedirs('processed_data', exist_ok=True)
    
    import pickle
    with open('processed_data/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
        
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
