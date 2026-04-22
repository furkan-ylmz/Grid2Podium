import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

# Sayfa ayarları
st.set_page_config(page_title="F1 Yarış Sonucu Tahmini", page_icon="🏎️", layout="centered")

# CSS ile karanlık tema ve şık görünüm
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .main-title {
        color: #FF1801; /* F1 Red */
        text-align: center;
        font-family: 'Helvetica', sans-serif;
        font-weight: bold;
        padding-top: 20px;
    }
    /* Girdi alanlarının başlıklarını beyaz yapalım */
    .stSelectbox label, .stNumberInput label {
        color: white !important;
    }
    .prediction-box {
        background-color: #1a1c24;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #FF1801;
        box-shadow: 0 4px 15px rgba(255, 24, 1, 0.2);
        text-align: center;
        margin-top: 25px;
    }
    /* Buton tasarımı */
    .stButton>button {
        background-color: #FF1801;
        color: white;
        border-radius: 5px;
        border: none;
        height: 3em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #D61400;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("<h1 class='main-title'>Formula 1 Sonuç Tahmin Sistemi 🏎️</h1>", unsafe_allow_html=True)
st.write("Eğitilmiş LSTM modelimizi kullanarak yarış sonuçlarını (Podyum, Puan veya Puansız) tahmin edin.")

# 1. Model Mimarisi Tanımı (train_models.py'deki ile aynı olmalı)
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        return self.fc(out)

# 2. Önbelleğe Alınan Dosyaları Yükleme Fonksiyonu
@st.cache_resource
def load_assets():
    with open('processed_data/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('models/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
        
    # Modeli başlatmak için feature_cols uzunluğunu alıyoruz
    input_dim = len(feature_cols)
    model = SimpleLSTM(input_dim=input_dim)
    model.load_state_dict(torch.load('models/lstm_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return encoders, feature_cols, model

try:
    encoders, feature_cols, model = load_assets()
except Exception as e:
    st.error(f"Gerekli dosyalar bulunamadı. Lütfen önce modelleri eğittiğinizden emin olun. Hata: {e}")
    st.stop()

# 3. Arayüz Girdileri (Inputs)
col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Sezon (Yıl)", options=range(2019, 2027), index=7)
    # Encoder'ların orijinal sınıflarını (string olan Track, Driver, vs.) seçenek olarak sunalım
    track = st.selectbox("Pist (Track)", options=sorted(encoders['Track'].classes_))
    starting_grid = st.number_input("Başlangıç Pozisyonu (Starting Grid)", min_value=1, max_value=24, value=1)

with col2:
    driver = st.selectbox("Sürücü (Driver)", options=sorted(encoders['Driver'].classes_))
    team = st.selectbox("Takım (Team)", options=sorted(encoders['Team'].classes_))

# 4. Tahmin Butonu
if st.button("🏎️ Sonucu Tahmin Et", use_container_width=True):
    with st.spinner('Motorlar ısınıyor... Yapay zeka düşünüyor...'):
        # Kullanıcı girdilerini sözlük olarak alalım
        # Encoder ile string seçimi -> sayıya çevirme
        track_encoded = encoders['Track'].transform([track])[0]
        driver_encoded = encoders['Driver'].transform([driver])[0]
        team_encoded = encoders['Team'].transform([team])[0]
        
        input_data = {
            'Track': track_encoded,
            'Driver': driver_encoded,
            'Team': team_encoded,
            'Starting Grid': starting_grid,
            'Year': year
        }
        
        # DataFrame oluştur
        input_df = pd.DataFrame([input_data])
        
        # Eğitimde olduğu gibi get_dummies yapalım
        cat_cols = ['Track', 'Driver', 'Team', 'Year']
        input_df = pd.get_dummies(input_df, columns=cat_cols)
        
        # Eğitimde beklenen tüm sütunları modelin beklediği sırayla oluştur
        # Eksik sütunlara 0 bas, fazla sütun varsa yoksay
        final_input = pd.DataFrame(columns=feature_cols)
        for col in feature_cols:
            if col in input_df.columns:
                final_input[col] = input_df[col]
            else:
                final_input[col] = 0
                
        # Tensor'a çevir
        X_tensor = torch.tensor(final_input.values.astype(np.float32))
        
        # Tahmin yap
        with torch.no_grad():
            output = model(X_tensor)
            _, predicted = torch.max(output.data, 1)
            predicted_class = predicted.item()
            
            # Olasılıkları da hesapla (Softmax ile)
            probabilities = torch.nn.functional.softmax(output, dim=1).numpy()[0]
            
        # Sonucu formatlama
        classes = {
            0: ("Podyum! (İlk 3)", "🏆", "#FFD700"),
            1: ("Puan Alır (İlk 10)", "✅", "#1E90FF"),
            2: ("Puan Alamaz / Bitiremez (11+)", "❌", "#A9A9A9")
        }
        
        text_result, icon, color = classes[predicted_class]
        
        # Sonucu şık bir kutu içinde göster
        st.markdown(f"""
            <div class='prediction-box'>
                <h2 style='color: {color}; margin: 0;'>{icon} {text_result}</h2>
                <br>
                <p><strong>Yapay Zeka Güven Oranı:</strong></p>
                <p>Podyum: %{probabilities[0]*100:.1f} | Puan: %{probabilities[1]*100:.1f} | Puansız: %{probabilities[2]*100:.1f}</p>
            </div>
        """, unsafe_allow_html=True)
