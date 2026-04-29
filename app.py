import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

st.set_page_config(page_title="F1 Yarış Sonucu Tahmini", page_icon="🏎️", layout="centered")

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

st.markdown("<h1 class='main-title'>Formula 1 Sonuç Tahmin Sistemi 🏎️</h1>", unsafe_allow_html=True)
st.write("En yüksek doğrulukla eğitilmiş yapay zeka modelimizi kullanarak yarış sonuçlarını (Podyum, Puan veya Puansız) tahmin edin.")

class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(CustomMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

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

class CNN1D(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            dummy = self.pool2(self.conv2(self.pool1(self.conv1(dummy))))
            flat_dim = dummy.view(1, -1).shape[1]
            
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, output_dim=3, d_model=64, nhead=4, num_layers=2):
        super(TabularTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1) # (batch, 1, d_model)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

@st.cache_resource
def load_assets():
    with open('processed_data/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('models/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    with open('models/best_model_arch.pkl', 'rb') as f:
        best_arch = pickle.load(f)

    input_dim = len(feature_cols)
    
    if best_arch == "CustomMLP":
        model = CustomMLP(input_dim=input_dim)
    elif best_arch == "CNN1D":
        model = CNN1D(input_dim=input_dim)
    elif best_arch == "TabularTransformer":
        model = TabularTransformer(input_dim=input_dim)
    else:
        model = SimpleLSTM(input_dim=input_dim)

    model.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return encoders, feature_cols, model, best_arch


try:
    encoders, feature_cols, model, best_arch = load_assets()
except Exception as e:
    st.error(f"Gerekli dosyalar bulunamadı. Lütfen önce modelleri eğittiğinizden emin olun. Hata: {e}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Sezon (Yıl)", options=range(2019, 2027), index=7)
    track = st.selectbox("Pist (Track)", options=sorted(encoders['Track'].classes_))
    starting_grid = st.number_input("Başlangıç Pozisyonu (Starting Grid)", min_value=1, max_value=24, value=1)

with col2:
    driver = st.selectbox("Sürücü (Driver)", options=sorted(encoders['Driver'].classes_))
    team = st.selectbox("Takım (Team)", options=sorted(encoders['Team'].classes_))

if st.button("🏎️ Sonucu Tahmin Et", use_container_width=True):
    with st.spinner('Motorlar ısınıyor... Yapay zeka düşünüyor...'):
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

        input_df = pd.DataFrame([input_data])

        cat_cols = ['Track', 'Driver', 'Team', 'Year']
        input_df = pd.get_dummies(input_df, columns=cat_cols)
        
        final_input = pd.DataFrame(columns=feature_cols)
        for col in feature_cols:
            if col in input_df.columns:
                final_input[col] = input_df[col]
            else:
                final_input[col] = 0

        X_numpy = final_input.values.astype(np.float32)

        X_tensor = torch.tensor(X_numpy)
        with torch.no_grad():
            output = model(X_tensor)
            _, predicted = torch.max(output.data, 1)
            predicted_class = predicted.item()
            probabilities = torch.nn.functional.softmax(output, dim=1).numpy()[0]

        classes = {
            0: ("Podyum! (İlk 3)", "🏆", "#FFD700"),
            1: ("Puan Alır (İlk 10)", "✅", "#1E90FF"),
            2: ("Puan Alamaz / Bitiremez (11+)", "❌", "#A9A9A9")
        }
        
        text_result, icon, color = classes[predicted_class]

        model_display_name = f"Modele Göre ({best_arch}) Tahmin:"

        st.markdown(f"""
            <div class='prediction-box'>
                <h2 style='color: {color}; margin: 0;'>{icon} {text_result}</h2>
                <br>
                <p><strong>{model_display_name} Güven Oranı:</strong></p>
                <p>Podyum: %{probabilities[0]*100:.1f} | Puan: %{probabilities[1]*100:.1f} | Puansız: %{probabilities[2]*100:.1f}</p>
            </div>
        """, unsafe_allow_html=True)
