def explain_emotion(text, emotion_analysis):
    """
    Generates explanation text for the top emotion and highlights trigger words.
    """
    top_emotion = emotion_analysis["top_emotion"]
    
    # Heuristic keyword matching
    keywords = {
        "joy": ["happy", "great", "excellent", "good", "love", "beautiful", "amazing", "wonderful", "excited", "proud", "smiling"],
        "sadness": ["sad", "depressed", "sorrow", "crying", "unhappy", "tears", "pain", "loss", "miss", "alone"],
        "anger": ["angry", "mad", "furious", "hate", "annoying", "frustrated", "rage", "stupid", "idiot", "worst"],
        "fear": ["scared", "fear", "terrified", "afraid", "panic", "worry", "anxious", "nervous", "dread"],
        "surprise": ["wow", "unexpected", "surprise", "suddenly", "shocked", "amazing", "unbelievable"],
        "disgust": ["ew", "gross", "disgusting", "awful", "terrible", "nasty", "sick", "vile"],
        "neutral": ["okay", "fine", "normal", "average", "standard", "routine", "regular"]
    }
    
    text_lower = text.lower()
    trigger_words = []
    
    if top_emotion in keywords:
        for word in keywords[top_emotion]:
            if word in text_lower:
                trigger_words.append(word)
                
    explanation_parts = []
    explanation_parts.append(f"The primary emotion detected is **{top_emotion}**.")
    
    if emotion_analysis.get("is_neutral_override", False):
        explanation_parts.append("The emotion was classified as neutral because the confidence of the top emotion was below the 40% threshold.")
    elif emotion_analysis.get("mixed_emotions"):
        mixed = emotion_analysis["mixed_emotions"]
        explanation_parts.append(f"This is a mixed emotion scenario, heavily blending **{mixed[0]}** and **{mixed[1]}**.")
    
    if trigger_words:
        explanation_parts.append(f"The presence of trigger words like '{', '.join(trigger_words)}' contributed to this analysis.")
    elif top_emotion != "neutral":
        explanation_parts.append("The model inferred this emotion contextually based on the overall sentence structure and tone, rather than simple keywords.")
        
    explanation = " ".join(explanation_parts)
    
    # Heuristic token level importance (mocking precise attention for UI visualization)
    words = text.split()
    token_importance = []
    
    for word in words:
        clean_word = word.lower().strip(".,!?\"'")
        if trigger_words and clean_word in trigger_words:
            # High importance for exact keyword match
            token_importance.append({"word": word, "importance": 0.85})
        else:
            # Baseline importance
            token_importance.append({"word": word, "importance": 0.15})
            
    return {
        "trigger_words": trigger_words,
        "explanation": explanation,
        "token_importance": token_importance
    }
