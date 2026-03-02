import streamlit as st
from emotion_model import load_model, predict_emotion
from utils import interpret_emotion

st.set_page_config(page_title="Advanced Emotion & Hate Speech Detector", layout="wide")

st.title("🚀 Advanced Emotion & Hate Speech Detection System")

@st.cache_resource
def initialize():
    return load_model()

tokenizer, model = initialize()

text = st.text_area("Enter your sentence:", height=150)

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        results = predict_emotion(text, tokenizer, model)
        label, confidence = interpret_emotion(results)

        st.subheader("🧠 Emotion Analysis")
        st.success(f"Top Emotion: {label}")
        st.write(f"Confidence: {confidence}%")

        st.subheader("📊 Full Emotion Probabilities")

        for emotion, prob in results.items():
            st.write(f"{emotion}: {round(prob * 100, 2)}%")

        st.bar_chart(results)