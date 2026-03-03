import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.request
import urllib.parse
import json
from langdetect import detect, DetectorFactory
import PyPDF2
import io

# Seed for langdetect
DetectorFactory.seed = 0

from emotion_model import load_model, predict_emotion
from hate_model import load_hate_model, predict_toxicity
from explainability import explain_emotion
from timeline import add_to_timeline, calculate_volatility, get_stability_metric, plot_emotional_timeline, plot_emotion_shifts

st.set_page_config(page_title="Emotion Intelligence Platform", layout="wide", page_icon="🧠")

# Custom CSS for UI Enhancement
st.markdown("""
<style>
    .emotion-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        color: #333;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .hate-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff3cd;
        color: #856404;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #ffeeba;
    }
    .hate-card.high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Emotion Intelligence Platform")
st.markdown("Analyze the meaning, context, implied emotions, and potential toxicity of text and documents.")

@st.cache_resource
def initialize_emotion_model():
    return load_model()

@st.cache_resource
def initialize_hate_model():
    return load_hate_model()

with st.spinner("Loading models... This may take a moment."):
    emo_tokenizer, emo_model = initialize_emotion_model()
    hate_tokenizer, hate_model = initialize_hate_model()

def translate_to_english(text):
    """Fallback translation using MyMemory API"""
    try:
        url = "https://api.mymemory.translated.net/get?q=" + urllib.parse.quote(text) + "&langpair=Autodetect|en"
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode("utf-8"))
        return data["responseData"]["translatedText"]
    except Exception as e:
        st.warning(f"Translation failed: {e}. Analyzing original text.")
        return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    return ""

def create_gauge_chart(confidence, title="Confidence"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "green"}
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    return fig

# Inputs
input_col, file_col = st.columns(2)

with input_col:
    text_input = st.text_area("Enter your sentence for analysis:", height=150)

with file_col:
    uploaded_file = st.file_uploader("Or upload a Document (.txt, .pdf)", type=["txt", "pdf"])

analyze_pressed = st.button("🔍 Analyze Content", type="primary")

if analyze_pressed:
    text_to_analyze = ""
    
    if uploaded_file is not None:
        text_to_analyze = extract_text_from_file(uploaded_file)
        st.info("Loaded text from file.")
    elif text_input.strip() != "":
        text_to_analyze = text_input.strip()
    else:
        st.warning("Please enter a sentence or upload a file.")
        st.stop()
        
    # Language Detection & Translation
    try:
        lang = detect(text_to_analyze)
        if lang != "en":
            st.info(f"Detected language: **{lang}**. Translating to English for analysis...")
            translated_text = translate_to_english(text_to_analyze)
            text_to_analyze = translated_text
            with st.expander("Show Translated Text"):
                st.write(translated_text)
    except Exception as e:
        pass # langdetect fails on numbers/empty strings
        
    # Analyze
    with st.spinner("Analyzing emotions and toxicity..."):
        emo_results = predict_emotion(text_to_analyze, emo_tokenizer, emo_model)
        hate_results = predict_toxicity(text_to_analyze, hate_tokenizer, hate_model)
        explanation_data = explain_emotion(text_to_analyze, emo_results)
        
        # Add to timeline
        add_to_timeline(text_to_analyze, emo_results)

    st.markdown("---")
    
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🧠 Emotion Analysis")
        
        # Determine confidence interpretation
        conf = emo_results["confidence"]
        if conf > 0.8:
            conf_interp = "High confidence"
        elif conf >= 0.5:
            conf_interp = "Moderate confidence"
        else:
            conf_interp = "Emotional ambiguity detected"
            
        top_emo = emo_results["top_emotion"].capitalize()
        if top_emo == "Neutral":
            emoji = "😐"
        elif top_emo in ["Joy", "Excitement", "Pride", "Relief", "Gratitude", "Love"]:
            emoji = "😊"
        elif top_emo in ["Sadness", "Disappointment"]:
            emoji = "😢"
        elif top_emo in ["Anger", "Disgust"]:
            emoji = "😠"
        elif top_emo in ["Fear", "Anxiety"]:
            emoji = "😨"
        else:
            emoji = "🤔"
            
        st.markdown(f"""
        <div class="emotion-card">
            <h4>{emoji} Primary Emotion: <strong>{top_emo}</strong></h4>
            <p>Confidence: {conf*100:.1f}% ({conf_interp})</p>
            <p><i>{explanation_data['explanation']}</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        if emo_results["mixed_emotions"]:
            st.warning(f"🎭 Mixed Emotion Detected: {emo_results['mixed_emotions'][0]} intertwined with {emo_results['mixed_emotions'][1]}")
            
        # Full Distribution Chart
        probs = emo_results["probabilities"]
        df_probs = pd.DataFrame(list(probs.items()), columns=["Emotion", "Probability"])
        fig_bar = px.bar(df_probs, x="Emotion", y="Probability", title="Full Emotion Distribution")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("🛡️ Hate Speech & Toxicity")
        
        risk_level = hate_results["risk_level"]
        tox_score = hate_results["toxicity_score"]
        
        card_class = "hate-card high-risk" if risk_level == "High" else "hate-card"
        icon = "⚠️" if risk_level == "High" else ("⚠️" if risk_level == "Medium" else "✅")
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4>{icon} Overall Risk Level: <strong>{risk_level}</strong></h4>
            <p>Maximum Toxicity Score: {tox_score*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(create_gauge_chart(tox_score, "Toxicity Gauge"), use_container_width=True)
        
        st.write("**Violation Probabilities:**")
        cols = st.columns(3)
        h_probs = list(hate_results["probabilities"].items())
        for i, (cat, p) in enumerate(h_probs):
            cols[i%3].metric(cat.capitalize(), f"{p*100:.1f}%")

    st.markdown("---")
    
    st.subheader("📈 Session Timeline Analytics")
    col_tl1, col_tl2 = st.columns(2)
    with col_tl1:
        st.metric("Emotional Volatility Score", f"{calculate_volatility():.1f}%", help="Frequency of changes in dominant emotion")
    with col_tl2:
        st.metric("Emotional Stability Score", f"{get_stability_metric():.1f}%", help="100% - Volatility")
        
    timeline_fig = plot_emotional_timeline()
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)

    st.markdown("---")
    
    # Advanced Mode Toggle
    st.subheader("⚙️ Advanced Analysis Mode")
    if st.checkbox("Enable Advanced Analysis"):
        st.write("### Raw Logits")
        st.json(emo_results["logits"])
        
        st.write("### Top 3 Emotions")
        top3 = list(emo_results["probabilities"].items())[:3]
        for e, p in top3:
            st.write(f"- **{e}**: {p*100:.2f}%")
            
        st.write("### Token Importance (Heuristic Extraction)")
        token_df = pd.DataFrame(explanation_data["token_importance"])
        st.dataframe(token_df, use_container_width=True)