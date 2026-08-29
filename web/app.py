import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
from core.predictor import RacePredictor

RESULTS_DIR = BASE_DIR / "results"

st.set_page_config(
    page_title="F1 Yarış Sonucu Tahmini",
    page_icon="🏎️",
    layout="centered"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .main-title {
        color: #FF1801; 
        text-align: center;
        font-family: 'Helvetica', sans-serif;
        font-weight: bold;
        padding-top: 20px;
    }
    .sub-title {
        text-align: center;
        color: #cccccc;
        font-size: 1.1em;
        margin-bottom: 25px;
    }
    .stSelectbox label, .stNumberInput label {
        color: white !important;
        font-weight: 500;
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
    .stButton>button {
        background-color: #FF1801;
        color: white;
        border-radius: 6px;
        border: none;
        height: 3.2em;
        font-weight: bold;
        font-size: 1.05em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #D61400;
        color: white;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Formula 1 Sonuç Tahmin Sistemi 🏎️</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>En yüksek doğrulukla eğitilmiş yapay zeka modelini kullanarak yarış sonuçlarını tahmin edin.</p>", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Cache and load predictor instance with best trained model and encoders."""
    return RacePredictor()


try:
    predictor = get_predictor()
    encoders = predictor.encoders
except Exception as e:
    st.error(f"Gerekli model veya veri dosyaları bulunamadı. Lütfen önce modelleri eğittiğinizden emin olun.\nHata: {e}")
    st.stop()

tab_predict, tab_metrics = st.tabs(["🎯 Sonuç Tahmini", "📊 Model Performans Analizi"])

with tab_predict:
    col1, col2 = st.columns(2)

    with col1:
        year = st.selectbox("Sezon (Yıl)", options=list(range(2019, 2027)), index=7)
        track = st.selectbox("Pist (Track)", options=sorted(encoders['Track'].classes_))
        starting_grid = st.number_input("Başlangıç Pozisyonu (Starting Grid)", min_value=1, max_value=24, value=1)

    with col2:
        driver = st.selectbox("Sürücü (Driver)", options=sorted(encoders['Driver'].classes_))
        team = st.selectbox("Takım (Team)", options=sorted(encoders['Team'].classes_))

    if st.button("🏎️ Sonucu Tahmin Et", width="stretch"):
        with st.spinner('Motorlar ısınıyor... Yapay zeka düşünüyor...'):
            try:
                result = predictor.predict(
                    track=track,
                    driver=driver,
                    team=team,
                    starting_grid=starting_grid,
                    year=year
                )

                st.markdown(f"""
                    <div class='prediction-box'>
                        <h2 style='color: {result["color"]}; margin: 0;'>{result["icon"]} {result["label"]}</h2>
                        <br>
                        <p><strong>Modele Göre ({result["best_arch"]}) Tahmin Güven Oranı:</strong></p>
                        <p>🏆 Podyum: %{result["probabilities"]["podium"]*100:.1f} | 
                           ✅ Puan: %{result["probabilities"]["points"]*100:.1f} | 
                           ❌ Puansız: %{result["probabilities"]["no_points"]*100:.1f}</p>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as ex:
                st.error(f"Tahmin sırasında bir hata oluştu: {ex}")

with tab_metrics:
    st.subheader("Modellerin Karşılaştırmalı Performans Raporu")
    metrics_img = RESULTS_DIR / "model_evaluation_metrics.png"
    if metrics_img.exists():
        st.image(str(metrics_img), caption="Model Değerlendirme Metrikleri", width="stretch")

    loss_img = RESULTS_DIR / "loss_curves.png"
    acc_img = RESULTS_DIR / "accuracy_curves.png"
    cm_img = RESULTS_DIR / "test_confusion_matrices.png"

    if loss_img.exists():
        st.subheader("Eğitim ve Doğrulama Kayıp (Loss) Grafikleri")
        st.image(str(loss_img), width="stretch")

    if acc_img.exists():
        st.subheader("Doğruluk (Accuracy) Grafikleri")
        st.image(str(acc_img), width="stretch")

    if cm_img.exists():
        st.subheader("Test Seti Konfüzyon Matrisleri")
        st.image(str(cm_img), width="stretch")
