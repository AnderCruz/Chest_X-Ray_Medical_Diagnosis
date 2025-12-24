import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile
import os
import requests

from inference import hospital_inference
from logger import log_inference

# ===============================
# CONFIGURAÇÃO DA PÁGINA
# ===============================
st.set_page_config(
    page_title="Pneumonia X-Ray AI",
    layout="centered"
)

st.title("🩺 Sistema de Apoio ao Diagnóstico — Pneumonia")

st.markdown("""
⚠️ **Aviso Clínico**  
Este sistema é um *apoio à decisão médica* e **não substitui avaliação clínica**.
""")

# ===============================
# CARREGAMENTO DO ARQUIVO
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_clinical.keras")
MODEL_URL = "https://drive.google.com/file/d/1R3rX15h_ARpFk-EPzuUrNGlzISJ9Vqy_"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Baixando modelo clínico..."):
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# ===============================
# PARÂMETROS CLÍNICOS
# ===============================
st.sidebar.header("⚙️ Configurações Clínicas")

threshold = st.sidebar.slider(
    "Threshold clínico (sensibilidade ↑)",
    min_value=0.80,
    max_value=0.99,
    value=0.97,
    step=0.01
)

model_version = "1.0.0"

# ===============================
# UPLOAD DA IMAGEM
# ===============================
uploaded_file = st.file_uploader(
    "📤 Envie a radiografia de tórax",
    type=["png", "jpg", "jpeg"]
)

# ===============================
# INFERÊNCIA
# ===============================
if uploaded_file is not None:
    st.image(uploaded_file, caption="Radiografia enviada", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if st.button("🔍 Analisar imagem"):
        with st.spinner("Analisando radiografia..."):
            result = hospital_inference(
                model=model,
                img_path=tmp_path,
                threshold=threshold,
                model_version=model_version
            )

            log_inference(result)

        os.remove(tmp_path)

        # ===============================
        # RESULTADO CLÍNICO
        # ===============================
        st.subheader("📊 Resultado da Análise")

        if result["prediction_label"] == 1:
            st.error("🟥 **PNEUMONIA DETECTADA**")
        else:
            st.success("🟩 **SEM SINAIS DE PNEUMONIA**")

        st.metric(
            label="Probabilidade de Pneumonia",
            value=f"{result['probability_pneumonia'] * 100:.2f}%"
        )

        st.markdown(f"""
**Recomendação Clínica:**  
{result["recommendation"]}

**Modelo:** {result["model_version"]}  
**Threshold Clínico:** {result["clinical_threshold"]}
        """)

        st.expander("📄 Detalhes técnicos").json(result)


