import os

import pandas as pd
import streamlit as st
import torch

from phishing_utils import (
    MODELS_DIR,
    create_model,
    encode_dataframe,
    load_pickle,
    prepare_email_dataframe,
)


st.set_page_config(page_title="Phishing Email Detection", page_icon="SH", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f5f7fa;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2a2f3a;
        background: #151a23;
        margin-top: 18px;
    }
    .safe-box {
        border-color: #1db954;
        box-shadow: 0 0 0 1px rgba(29, 185, 84, 0.18);
    }
    .phishing-box {
        border-color: #ff4b4b;
        box-shadow: 0 0 0 1px rgba(255, 75, 75, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Phishing Email Detection System")
st.caption("Best deep learning model trained on the MeAJOR phishing email dataset.")


@st.cache_resource
def load_assets():
    assets_path = os.path.join(MODELS_DIR, "phishing_assets.pkl")
    model_path = os.path.join(MODELS_DIR, "best_phishing_model.pth")

    if not os.path.exists(assets_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model assets not found. Run data_preprocessing.py and train_models.py first."
        )

    assets = load_pickle(assets_path)
    model = create_model(
        model_key=assets["best_model_key"],
        vocab_size=len(assets["vocabulary"]),
        numeric_dim=len(assets["numeric_feature_columns"]),
        output_dim=2,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return assets, model


try:
    assets, model = load_assets()
except Exception as exc:
    st.error(str(exc))
    st.stop()


col_left, col_right = st.columns([2, 1])

with col_left:
    subject = st.text_input("Email subject", placeholder="Urgent: Verify your account")
    body = st.text_area(
        "Email body",
        height=280,
        placeholder="Paste the email content here...",
    )

with col_right:
    url_count = st.number_input("URL count", min_value=0.0, value=0.0, step=1.0)
    url_length_max = st.number_input("Max URL length", min_value=0.0, value=0.0, step=1.0)
    url_length_avg = st.number_input("Average URL length", min_value=0.0, value=0.0, step=1.0)
    url_subdom_max = st.number_input("Max subdomain count", min_value=0.0, value=0.0, step=1.0)
    url_subdom_avg = st.number_input("Average subdomain count", min_value=0.0, value=0.0, step=1.0)
    attachment_count = st.number_input("Attachment count", min_value=0.0, value=0.0, step=1.0)
    has_attachments = st.checkbox("Has attachments")
    content_types = st.selectbox(
        "Content type",
        options=[
            "text/plain",
            "text/html",
            "multipart/alternative",
            "multipart/mixed",
            "other",
        ],
        index=0,
    )
    language = st.selectbox("Language", options=["en", "de", "fr", "es", "other"], index=0)


if st.button("Predict Email Type", use_container_width=True):
    if not subject.strip() and not body.strip():
        st.warning("Please provide at least a subject or body.")
        st.stop()

    input_df = pd.DataFrame(
        [
            {
                "subject": subject,
                "body": body,
                "url_count": url_count,
                "url_length_max": url_length_max,
                "url_length_avg": url_length_avg,
                "url_subdom_max": url_subdom_max,
                "url_subdom_avg": url_subdom_avg,
                "attachment_count": attachment_count,
                "has_attachments": has_attachments,
                "content_types": content_types if content_types != "other" else "",
                "language": language if language != "other" else "",
            }
        ]
    )

    prepared_df = prepare_email_dataframe(input_df)
    text_sequences, numeric_features, _ = encode_dataframe(
        prepared_df,
        assets["vocabulary"],
        assets["numeric_scaler"],
        assets["max_length"],
    )

    text_tensor = torch.tensor(text_sequences, dtype=torch.long)
    numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(text_tensor, numeric_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(torch.argmax(logits, dim=1).item())

    safe_prob = probabilities[0] * 100
    phishing_prob = probabilities[1] * 100

    if predicted_class == 1:
        css_class = "result-box phishing-box"
        label = "Phishing Email"
        color = "#ff8080"
    else:
        css_class = "result-box safe-box"
        label = "Safe Email"
        color = "#7fe7a5"

    st.markdown(
        f"""
        <div class="{css_class}">
            <h2 style="margin:0; color:{color};">{label}</h2>
            <p style="margin-top:10px;">Model: {assets["best_model_name"]}</p>
            <p>Safe probability: %{safe_prob:.2f}</p>
            <p>Phishing probability: %{phishing_prob:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
