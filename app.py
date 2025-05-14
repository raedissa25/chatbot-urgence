import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from docx import Document
import google.generativeai as genai
from tensorflow.keras.models import load_model

# ---------- Configuration Gemini ----------
api_key = st.secrets["GEMINI_API_KEY"]
if not api_key:
    st.error("Clé API Gemini manquante dans secrets.toml")
    st.stop()

genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# ---------- Charger le modèle CNN ----------
MODEL_PATH = "CNN 1D model.h5"
try:
    cnn_model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

# ---------- Dictionnaire des classes ----------
classes = {
    0: ("Normal (N)", "Battement cardiaque normal (rythme sinusal normal)."),
    1: ("Extrasystole ventriculaire (VEB / PVC)", "Battement prématuré provenant des ventricules."),
    2: ("Battements supraventriculaires (SVEB)", "Battements prématurés provenant des oreillettes."),
    3: ("Fusion de battements (F)", "Fusion entre un battement normal et un VEB."),
    4: ("Battements inconnus (Q)", "Battements non classifiés dans les catégories précédentes."),
}

# ---------- Fonctions ----------
def preprocess_ecg(csv_file):
    df = pd.read_csv(csv_file, header=None)
    if df.shape[1] != 188:
        st.error("⚠️ Le fichier CSV doit contenir exactement 188 colonnes.")
        return None
    ecg_signal = df.iloc[:, :-1].values
    ecg_data = np.expand_dims(ecg_signal, axis=-1)
    return ecg_data

def plot_ecg_signal(signal, title="📈 Signal ECG (1er échantillon)"):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal, color='dodgerblue')
    ax.set_title(title)
    ax.set_xlabel("Temps (échantillons)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

# Nouveau : réponse dans la langue de la question
def get_bot_response(user_input):
    prompt = f"""
You are a professional, empathetic, and kind-hearted cardiologist.

Your role is to help users understand:

Electrocardiograms (ECGs)

Heart diseases

Cardiovascular health

Heart-related treatments

Always reply in the same language used in the user's question.

If the question is not related to cardiology, politely respond (in the same language):
"I specialize in cardiology. I'm here to help you only with matters related to the heart and cardiovascular health."

Use a clear, human, reassuring, and professional tone.
"""

    chat_session = model.start_chat(history=[
        {"role": "user", "parts": [prompt]}
    ])
    response = chat_session.send_message(user_input)
    return response.text

def generate_word_report(class_name, class_description, prediction):
    doc = Document()
    doc.add_heading("Rapport d’Analyse ECG", level=1)

    doc.add_heading("Résultat Principal :", level=2)
    doc.add_paragraph(f"Classe prédite : {class_name}")
    doc.add_paragraph(f"Description : {class_description}")

    doc.add_heading("Détail des probabilités :", level=2)
    for idx, prob in enumerate(prediction[0]):
        name, desc = classes[idx]
        doc.add_paragraph(f"{name} ({desc}) : {prob:.4f}")

    file_stream = BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream

# ---------- Application principale ----------
def main():
    st.set_page_config(page_title="Assistant Médical Cardiaque", page_icon="🫀")
    st.title("🫀 Assistant Médical Cardiaque")

    st.header("📂 Téléversez un fichier ECG (CSV)")
    uploaded_file = st.file_uploader("Choisissez un fichier .csv", type=["csv"])

    if uploaded_file is not None:
        st.success("✅ Fichier uploadé avec succès !")
        ecg_input = preprocess_ecg(uploaded_file)

        if ecg_input is not None:
            st.subheader("📊 Signal ECG détecté")
            plot_ecg_signal(ecg_input[0].squeeze())

            prediction = cnn_model.predict(ecg_input)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            class_name, class_description = classes[predicted_class_idx]

            st.subheader("🔍 Résultat de l’analyse automatique :")
            st.success(f"Classe prédite : **{class_name}**")
            st.markdown(f"📝 *{class_description}*")

            st.subheader("📊 Détail des probabilités par classe :")
            for idx, prob in enumerate(prediction[0]):
                name, desc = classes[idx]
                st.markdown(f"**{name}** — {desc} : **{prob:.4f}**")
                st.progress(float(prob))

            st.subheader("📥 Télécharger le rapport médical (.docx)")
            word_file = generate_word_report(class_name, class_description, prediction)
            st.download_button(
                label="📄 Télécharger le rapport Word",
                data=word_file,
                file_name="rapport_ecg.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    # Chat IA
    st.header("💬 Posez une question à l’IA cardiologue")
    st.markdown("🩺 *Je suis un assistant médical spécialisé en **cardiologie**. Posez-moi vos questions sur le cœur, l’ECG, ou les traitements cardiaques.*")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Posez une question sur l’ECG, les anomalies, etc.")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Réflexion en cours..."):
            bot_response = get_bot_response(user_input)

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)

if __name__ == "__main__":
    main()
