import streamlit as st
import pandas as pd
import plotly.express as px

def init_timeline():
    if "emotion_history" not in st.session_state:
        st.session_state["emotion_history"] = []

def add_to_timeline(text, emotion_analysis):
    """
    Append an analysis result to the history timeline.
    """
    init_timeline()
    
    # We store basic details and all probabilities for plotting.
    entry = {
        "text": text,
        "top_emotion": emotion_analysis["top_emotion"],
        "confidence": emotion_analysis["confidence"]
    }
    
    # Flatten probabilities into the entry
    if "probabilities" in emotion_analysis:
        for emotion, prob in emotion_analysis["probabilities"].items():
            entry[f"{emotion}_score"] = prob
            
    st.session_state["emotion_history"].append(entry)

def calculate_volatility():
    """
    Calculate emotional volatility: the percentage of times the top emotion changes.
    """
    init_timeline()
    history = st.session_state["emotion_history"]
    
    if not history or len(history) < 2:
        return 0.0
        
    changes = 0
    for i in range(1, len(history)):
        if history[i]["top_emotion"] != history[i-1]["top_emotion"]:
            changes += 1
            
    volatility = changes / (len(history) - 1)
    return volatility * 100 # percentage

def get_stability_metric():
    """
    Calculates stability: 100% - volatility
    """
    return 100.0 - calculate_volatility()

def plot_emotional_timeline():
    """
    Returns a Plotly Figure showing confidence of the dominant emotion over time.
    """
    init_timeline()
    history = st.session_state["emotion_history"]
    
    if not history:
        return None
        
    df = pd.DataFrame(history)
    df["Entry Index"] = df.index + 1
    
    fig = px.line(df, x="Entry Index", y="confidence", color="top_emotion", 
                  markers=True, title="Dominant Emotion Confidence Timeline",
                  hover_data=["text"])
                  
    fig.update_layout(yaxis_range=[0, 1.1])
    return fig

def plot_emotion_shifts():
    """
    Plots a multi-line graph of all tracked emotion scores over time.
    """
    init_timeline()
    history = st.session_state["emotion_history"]
    
    if not history:
        return None
        
    df = pd.DataFrame(history)
    df["Entry Index"] = df.index + 1
    
    # Extract only emotion score columns to melt
    score_cols = [c for c in df.columns if c.endswith("_score")]
    if not score_cols:
        return None
        
    df_melt = pd.melt(df, id_vars=["Entry Index", "text"], value_vars=score_cols,
                      var_name="Emotion", value_name="Probability")
    df_melt["Emotion"] = df_melt["Emotion"].apply(lambda x: x.replace("_score", ""))
    
    fig = px.line(df_melt, x="Entry Index", y="Probability", color="Emotion",
                  title="Probability Shifts Over Time", markers=True, 
                  hover_data=["text"])
    fig.update_layout(yaxis_range=[0, 1.1])
    return fig
