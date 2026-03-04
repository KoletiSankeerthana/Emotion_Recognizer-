import streamlit as st
import pandas as pd
import plotly.express as px
from emotion_model import load_model, predict_emotion
from hate_model import load_hate_model, predict_toxicity

import json

st.set_page_config(page_title="Emotion Intelligence System", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="color: #2c3e50; font-family: 'Helvetica Neue', sans-serif;">Emotion Intelligence System</h1>
    <h3 style="color: #7f8c8d; font-family: 'Helvetica Neue', sans-serif; font-weight: 300;">AI-powered Emotion and Hate Speech Analysis</h3>
    <p style="color: #95a5a6; max-width: 600px; margin: 0 auto; line-height: 1.5;">
        This tool analyzes emotional tone, detects mixed emotional states, and identifies toxic language using transformer-based NLP models.
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
    
    full_res = emo_results["full_sentence"]
    top_emo = full_res["top_emotion"]
    
    # --- UI LAYOUT WITH MAIN COLUMNS ---
    st.markdown("### 🧠 Emotion Analysis")
    
    # Emotion Cards: Left summary, Right metrics
    e_col1, e_col2 = st.columns([1, 1], gap="large")
    
    with e_col1:
        html_content = f"""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef;'>
        """
        
        if "Mixed" in top_emo:
            html_content += f"<h4 style='margin-bottom: 5px;'>Primary Emotion: <b>{full_res['primary'].split(' ')[0].title()}</b></h4>"
            html_content += f"<h4 style='margin-bottom: 10px;'>Secondary Emotion: <b>{full_res['secondary'].split(' ')[0].title()}</b></h4>"
        else:
            html_content += f"<h4 style='margin-bottom: 10px;'>Primary Emotion: <b>{top_emo.title()}</b></h4>"
            
        html_content += f"<div style='background-color: #ffffff; padding: 15px; border-left: 4px solid #3498db; margin-top: 15px;'>"
        html_content += f"<p style='margin-bottom: 5px; color: #7f8c8d; font-size: 0.9em; text-transform: uppercase;'><b>Interpretation</b></p>"
        html_content += f"<p style='color: #2c3e50; font-size: 1.0em; margin-bottom: 0;'>{full_res.get('interpretation', '')}</p>"
        html_content += "</div>"
            
        html_content += "</div>"
        st.markdown(html_content, unsafe_allow_html=True)

    with e_col2:
        # 3 aligned Metric cards
        st.markdown("#### Emotional Metrics")
        
        conf_val = float(full_res['primary'].split(' ')[1].strip('()%')) / 100.0 if 'Mixed' not in top_emo else float(full_res['primary'].split(' ')[1].strip('()%')) / 100.0
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Confidence", f"{conf_val*100:.0f}%")
        m_col2.metric("Dominance", f"{full_res['dominance']:.2f}")
        m_col3.metric("Balance Index", f"{full_res['balance']:.2f}")
        
        st.markdown("<div style='font-size: 0.9em; color: #555; margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {full_res.get('reliability', 'High confident prediction.')}")
        st.markdown(f"**Dominance:** {full_res.get('dominance_text', 'Emotion strongly dominant.')}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("#### Emotional Polarity")
        p_col1, p_col2 = st.columns(2)
        pos = full_res.get('pos_score', 0)
        neg = full_res.get('neg_score', 0)
        p_col1.metric("Positive Emotion", f"{pos*100:.0f}%")
        p_col2.metric("Negative Emotion", f"{neg*100:.0f}%")
        st.markdown(f"**Overall Tone:** {full_res['sentiment']}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Clause Breakdown
    if len(emo_results["clauses"]) > 1:
        st.markdown("#### Clause Analysis")
        for i, c_data in enumerate(emo_results["clauses"]):
            c_res = c_data["results"]
            c_top_str = c_res["primary"] if "Mixed" not in c_res["top_emotion"] else f"{c_res['primary']} & {c_res['secondary']}"
            st.markdown(f"**Clause {i+1}** → {c_top_str}")
        
        final_emo = full_res['primary'].split(' ')[0].title() if "Mixed" not in top_emo else "Mixed Emotion"
        st.markdown(f"**Final Result** → {final_emo}")
            
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

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Triggers and Insights
    triggers = full_res.get("trigger_words", [])
    if triggers:
        st.markdown("#### Emotional Triggers")
        import re
        highlighted_text = text_input
        for t in triggers:
            # Add simple bolding for triggers
            highlighted_text = re.sub(rf"(?i)\b({re.escape(t)})\b", r"**\1**", highlighted_text)
            
        st.markdown(f"> {highlighted_text}")
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Suggested Insight")
    st.info(f"{full_res['suggestion']}")
    
    st.markdown("---")

    # Emotion Distribution Chart
    st.markdown("### 📊 Emotion Visualization")
    df_emo = pd.DataFrame(list(full_res["probabilities"].items()), columns=["Emotion", "Probability"])
    
    # Consistent subtle color mapping based on sentiment
    pos_emotions = ["joy", "love", "optimism", "pride", "gratitude", "excitement"]
    neg_emotions = ["sadness", "anger", "fear", "disgust", "disappointment", "anxiety"]
    
    def get_color(row):
        is_top = row['Emotion'] == top_emo.split(' ')[0].lower() if 'Mixed' not in top_emo else row['Emotion'] in [full_res['primary'].split(' ')[0].lower(), full_res['secondary'].split(' ')[0].lower()]
        if is_top: return '#3498db' # distinct blue for top emotion
        if row['Emotion'] in pos_emotions: return '#82e0aa' # lighter green
        elif row['Emotion'] in neg_emotions: return '#f5b7b1' # lighter red
        else: return '#e5e7eb' # flat grey
        
    df_emo['Color'] = df_emo.apply(get_color, axis=1)
    df_emo['Prob_Text'] = df_emo['Probability'].apply(lambda x: f"{x*100:.1f}%")
    df_emo = df_emo.sort_values(by="Probability", ascending=False)
    
    fig_emo = px.bar(
        df_emo, 
        x="Emotion", 
        y="Probability", 
        orientation='v',
        text="Prob_Text",
        color="Emotion",
        color_discrete_map={row['Emotion']: row['Color'] for _, row in df_emo.iterrows()}
    )
    fig_emo.update_traces(textposition='outside', width=0.4)
    fig_emo.update_layout(
        height=400, 
        showlegend=False, 
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor='#f8f9fa', range=[0, 1.1])
    )
    st.plotly_chart(fig_emo, use_container_width=True)
    
    st.divider()
    
    # Hate Speech Section
    st.markdown("### 🛡️ Hate Speech Analysis")
    top_hate = hate_results["top_category"]
    risk_level = hate_results.get("risk_level", "Low")
    explanation = hate_results.get("explanation", "")
    
    html_content_h = f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; margin-bottom: 20px;'>
        <h4 style='margin-bottom: 15px;'>Top Category: <b>{top_hate.capitalize()}</b></h4>
    """
    
    conf_val_h = hate_results['probabilities'].get(top_hate, 1.0)
    
    html_content_h += f"<p style='margin-bottom: 5px;'><i>Confidence: {conf_val_h*100:.0f}%</i></p>"
    html_content_h += f"<p style='margin-bottom: 10px;'>Hate Speech Risk Level: <b>{risk_level}</b></p>"
    
    html_content_h += f"<div style='background-color: #ffffff; padding: 15px; border-left: 4px solid {'#e74c3c' if risk_level != 'Low' else '#95a5a6'};'>"
    html_content_h += f"<p style='margin-bottom: 5px; color: #7f8c8d; font-size: 0.9em; text-transform: uppercase;'><b>Explanation</b></p>"
    html_content_h += f"<p style='color: #2c3e50; font-size: 1.0em; margin-bottom: 0;'>{explanation}</p>"
    html_content_h += "</div>"
        
    html_content_h += "</div>"
    st.markdown(html_content_h, unsafe_allow_html=True)
    
    df_hate = pd.DataFrame(list(hate_results["probabilities"].items()), columns=["Category", "Probability"])
    
    def get_hate_color(cat):
        return '#95a5a6' if cat == 'neutral' else '#e74c3c'
        
    df_hate['Color'] = df_hate['Category'].apply(get_hate_color)
    df_hate['Prob_Text'] = df_hate['Probability'].apply(lambda x: f"{x*100:.1f}%")
    df_hate = df_hate.sort_values(by="Probability", ascending=False)
    
    fig_hate = px.bar(
        df_hate, 
        x="Category", 
        y="Probability", 
        orientation='v',
        text="Prob_Text",
        color="Category",
        color_discrete_map={row['Category']: row['Color'] for _, row in df_hate.iterrows()}
    )
    fig_hate.update_traces(textposition='outside', width=0.2)
    fig_hate.update_layout(
        height=250, 
        showlegend=False, 
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[0, 1.1])
    )
    st.plotly_chart(fig_hate, use_container_width=True)

        
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #95a5a6; font-size: 0.9em; border-top: 1px solid #e9ecef; padding-top: 20px;">
    Built using Transformer-based NLP models and deployed with Streamlit.
</div>
""", unsafe_allow_html=True)