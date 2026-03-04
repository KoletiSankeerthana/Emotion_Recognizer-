import streamlit as st
import pandas as pd
import plotly.express as px
from emotion_model import load_model, predict_emotion
from hate_model import load_hate_model, predict_toxicity

st.set_page_config(page_title="Emotion & Hate Speech Detection", layout="centered")

st.title("Emotion & Hate Speech Detection")

@st.cache_resource
def initialize_emotion_model():
    return load_model()

@st.cache_resource
def initialize_hate_model():
    return load_hate_model()

with st.spinner("Loading models..."):
    emo_tokenizer, emo_model = initialize_emotion_model()
    hate_tokenizer, hate_model = initialize_hate_model()

text_input = st.text_area("Enter your sentence for analysis:", height=150)

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter a sentence.")
        st.stop()
        
    with st.spinner("Analyzing..."):
        emo_results = predict_emotion(text_input, emo_tokenizer, emo_model)
        hate_results = predict_toxicity(text_input, hate_tokenizer, hate_model)
        
    st.markdown("---")
    
    # Emotion Section
    st.subheader("Emotion Detection Section")
    full_res = emo_results["full_sentence"]
    top_emo = full_res["top_emotion"]
    
    # 1. Broad Sentiment & Core Emotion
    st.write(f"**Overall Sentiment:** {full_res['sentiment']}")
    st.write(f"**Top Emotion:** {top_emo.title()}")
    st.write(f"*{full_res['reliability']}*")
    
    if "Mixed" in top_emo:
        st.write(f"**Primary:** {full_res['primary']}")
        st.write(f"**Secondary:** {full_res['secondary']}")
        
    # 2. Emotional Conflict Detection
    if len(emo_results["clauses"]) > 1:
        st.write("---")
        st.write("**Clause Analysis:**")
        
        has_pos = False
        has_neg = False
        pos_conf = 0
        neg_conf = 0
        
        for i, c_data in enumerate(emo_results["clauses"]):
            c_res = c_data["results"]
            c_top = c_res["top_emotion"]
            c_clause = c_data["clause"]
            
            if "Mixed" not in c_top:
                c_top_str = c_res["primary"]
            else:
                c_top_str = f"{c_res['primary']}, {c_res['secondary']}"
                
            st.write(f"- Clause {i+1}: \"{c_clause}\"")
            st.write(f"  → {c_top_str}")
            
            if c_res["sentiment"] == "Positive":
                has_pos = True
                pos_conf = max(pos_conf, c_res["pos_score"])
            if c_res["sentiment"] == "Negative":
                has_neg = True
                neg_conf = max(neg_conf, c_res["neg_score"])
                
        if has_pos and has_neg:
            st.error("**Emotional Conflict Detected**")
            st.write(f"- Positive Confidence: {pos_conf*100:.1f}%")
            st.write(f"- Negative Confidence: {neg_conf*100:.1f}%")

    # 3. Dominance, Triggers & Suggestions
    st.write("---")
    
    st.write(f"**Softmax Entropy (Uncertainty):** {full_res.get('entropy', 0):.2f}")
    st.write(f"**Emotional Dominance Score:** {full_res['dominance']:.2f}")
    st.write(f"**Emotional Balance Index:** {full_res['balance']:.2f}")
    st.write(f"*{full_res['dominance_text']}*")
    
    triggers = full_res.get("trigger_words", [])
    if triggers:
        st.write("**Emotional Trigger:**")
        for t in triggers:
            st.write(f"- \"{t}\"")
            
    st.info(f"**Suggestion:** {full_res['suggestion']}")
    
    st.write("---")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**All Emotion Probabilities:**")
        for e, p in full_res["probabilities"].items():
            st.write(f"- {e.capitalize()}: {p*100:.1f}%")
            
    with col2:
        df_emo = pd.DataFrame(list(full_res["probabilities"].items()), columns=["Emotion", "Confidence (%)"])
        df_emo["Confidence (%)"] = df_emo["Confidence (%)"] * 100
        fig_emo = px.bar(df_emo, x="Emotion", y="Confidence (%)", title="Emotion Confidence Scores")
        fig_emo.update_layout(height=400)
        st.plotly_chart(fig_emo, use_container_width=True)
        
    st.markdown("---")
    
    # Hate Speech Section
    st.subheader("Hate Speech Detection Section")
    top_hate = hate_results["top_category"]
    st.write(f"**Top Hate Category:** {top_hate.capitalize()}")
    
    h_col1, h_col2 = st.columns([1, 2])
    
    with h_col1:
        st.write("**All Hate Probabilities:**")
        for h, p in hate_results["probabilities"].items():
            st.write(f"- {h.capitalize()}: {p*100:.1f}%")
            
    with h_col2:
        df_hate = pd.DataFrame(list(hate_results["probabilities"].items()), columns=["Category", "Confidence (%)"])
        df_hate["Confidence (%)"] = df_hate["Confidence (%)"] * 100
        fig_hate = px.bar(df_hate, x="Category", y="Confidence (%)", title="Hate Confidence Scores")
        fig_hate.update_layout(height=400)
        st.plotly_chart(fig_hate, use_container_width=True)