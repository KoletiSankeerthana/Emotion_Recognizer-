import streamlit as st
import pandas as pd
import plotly.express as px
from emotion_model import load_model, predict_emotion
from hate_model import load_hate_model, predict_toxicity

st.set_page_config(page_title="Emotion & Hate Speech Detection", layout="centered")

st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="color: #2c3e50; font-family: 'Helvetica Neue', sans-serif;">Emotion Intelligence System</h1>
    <h3 style="color: #7f8c8d; font-family: 'Helvetica Neue', sans-serif; font-weight: 300;">AI-powered Emotion and Hate Speech Analysis</h3>
    <p style="color: #95a5a6; max-width: 600px; margin: 0 auto; line-height: 1.5;">
        This tool analyzes text to detect emotional tone, mixed emotional states, and potential hate speech using transformer-based NLP models.
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_emotion_model():
    return load_model()

@st.cache_resource
def initialize_hate_model():
    return load_hate_model()

with st.spinner("Loading models..."):
    emo_tokenizer, emo_model = initialize_emotion_model()
    hate_tokenizer, hate_model = initialize_hate_model()

st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    text_input = st.text_area(
        "📝 Enter text for analysis:", 
        height=150, 
        placeholder="Enter a sentence to analyze emotional tone and potential toxicity...",
        max_chars=1000
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🔍 Analyze Content", use_container_width=True, type="primary")

if analyze_btn:
    if not text_input.strip():
        st.warning("Please enter a sentence.")
        st.stop()
        
    with st.spinner("Analyzing..."):
        emo_results = predict_emotion(text_input, emo_tokenizer, emo_model)
        hate_results = predict_toxicity(text_input, hate_tokenizer, hate_model)
        
    st.markdown("---")
    
    # --- UI LAYOUT WITH MAIN COLUMNS ---
    main_col1, main_col2 = st.columns([1, 1], gap="large")
    
    with main_col1:
        st.markdown("### 🧠 Emotion Analysis")
        st.markdown(f"**Overall Sentiment:** {full_res['sentiment']}")
        
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef;'>
        """, unsafe_allow_html=True)
        
        if "Mixed" in top_emo:
            st.markdown(f"#### Primary Emotion: **{full_res['primary'].split(' ')[0].title()}**")
            st.markdown(f"#### Secondary Emotion: **{full_res['secondary'].split(' ')[0].title()}**")
            st.markdown(f"*Confidence: {full_res['primary'].split(' ')[1].strip('()')}*")
            st.markdown(f"*{full_res['reliability']}*")
            st.markdown("<br>_The sentence expresses an emotional conflict or blend of multiple distinct feelings._", unsafe_allow_html=True)
        else:
            st.markdown(f"#### Primary Emotion: **{top_emo.title()}**")
            conf_val = float(full_res['primary'].split(' ')[1].strip('()%')) / 100.0
            st.markdown(f"*Confidence: {conf_val*100:.0f}%*")
            st.markdown(f"*{full_res['reliability']}*")
        
        st.markdown("</div><br>", unsafe_allow_html=True)

        if len(emo_results["clauses"]) > 1:
            st.markdown("#### Clause-Level Breakdown")
            for i, c_data in enumerate(emo_results["clauses"]):
                c_res = c_data["results"]
                c_top_str = c_res["primary"] if "Mixed" not in c_res["top_emotion"] else f"{c_res['primary']} & {c_res['secondary']}"
                st.markdown(f"- **Clause {i+1}**: → {c_top_str}")
                
            has_pos = False
            has_neg = False
            for c_data in emo_results["clauses"]:
                c_res = c_data["results"]
                if c_res["sentiment"] == "Positive": has_pos = True
                if c_res["sentiment"] == "Negative": has_neg = True
                    
            if has_pos and has_neg:
                st.markdown("""
                <div style='background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                ⚠️ <b>Emotional Conflict Detected</b>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>#### Emotional Metrics", unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Dominance Score", f"{full_res['dominance']:.2f}")
        m_col2.metric("Balance Index", f"{full_res['balance']:.2f}")
        st.caption(f"*Prediction is {full_res['dominance_text'].lower()}. Softmax Entropy: {full_res.get('entropy', 0):.2f}*")
        
        triggers = full_res.get("trigger_words", [])
        if triggers:
            st.markdown("**Emotional Triggers:**")
            st.markdown(", ".join([f"*{t}*" for t in triggers]))

        st.markdown("<br>#### Suggested Insight", unsafe_allow_html=True)
        st.info(f"{full_res['suggestion']}")

    with main_col2:
        st.markdown("### Emotion Distribution")
        df_emo = pd.DataFrame(list(full_res["probabilities"].items()), columns=["Emotion", "Probability"])
        
        # Consistent subtle color mapping based on sentiment
        pos_emotions = ["joy", "love", "optimism", "pride", "gratitude", "excitement"]
        neg_emotions = ["sadness", "anger", "fear", "disgust", "disappointment", "anxiety"]
        
        def get_color(row):
            if row['Emotion'] in pos_emotions: return '#2ecc71' # subtle green
            elif row['Emotion'] in neg_emotions: return '#e74c3c' # subtle red
            else: return '#95a5a6' # neutral grey
            
        df_emo['Color'] = df_emo.apply(get_color, axis=1)
        df_emo = df_emo.sort_values(by="Probability", ascending=True) # Plotly shows bottom to top
        df_emo = df_emo[df_emo["Probability"] > 0.01].tail(8) # Show top relevant
        
        fig_emo = px.bar(
            df_emo, 
            y="Emotion", 
            x="Probability", 
            orientation='h',
            color="Emotion",
            color_discrete_map={row['Emotion']: row['Color'] for _, row in df_emo.iterrows()}
        )
        fig_emo.update_layout(
            height=400, 
            showlegend=False, 
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[0, 1])
        )
        st.plotly_chart(fig_emo, use_container_width=True)
        
    st.divider()
    
    # Hate Speech Section
    st.markdown("### 🛡️ Hate Speech Analysis")
    top_hate = hate_results["top_category"]
    
    h_col1, h_col2 = st.columns([1, 1], gap="large")
    
    with h_col1:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef;'>
        """, unsafe_allow_html=True)
        
        st.markdown(f"#### Top Category: **{top_hate.capitalize()}**")
        if top_hate != "neutral":
            conf_val_h = hate_results['probabilities'][top_hate]
            st.markdown(f"*Confidence: {conf_val_h*100:.0f}%*")
            st.markdown("<br>_The model detected traces of aggressive or toxic language._", unsafe_allow_html=True)
        else:
            conf_val_h = hate_results['probabilities'].get('neutral', 1.0)
            st.markdown(f"*Confidence: {conf_val_h*100:.0f}%*")
            st.markdown("<br>_The text appears to be clean standard language._", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with h_col2:
        df_hate = pd.DataFrame(list(hate_results["probabilities"].items()), columns=["Category", "Probability"])
        
        def get_hate_color(cat):
            return '#95a5a6' if cat == 'neutral' else '#e74c3c'
            
        df_hate['Color'] = df_hate['Category'].apply(get_hate_color)
        df_hate = df_hate.sort_values(by="Probability", ascending=True)
        
        fig_hate = px.bar(
            df_hate, 
            y="Category", 
            x="Probability", 
            orientation='h',
            color="Category",
            color_discrete_map={row['Category']: row['Color'] for _, row in df_hate.iterrows()}
        )
        fig_hate.update_layout(
            height=250, 
            showlegend=False, 
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[0, 1])
        )
        st.plotly_chart(fig_hate, use_container_width=True)
        
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #95a5a6; font-size: 0.9em; border-top: 1px solid #e9ecef; padding-top: 20px;">
    Built using Transformer-based NLP models and deployed with Streamlit.
</div>
""", unsafe_allow_html=True)