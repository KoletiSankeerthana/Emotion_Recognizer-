def process_emotions(raw_results):
    """
    Map 28 GoEmotions to target 13 categories:
    joy, sadness, anger, fear, surprise, disgust, love, optimism, pride,
    gratitude, disappointment, anxiety, neutral.
    """
    targets = {
        "joy": ["joy", "amusement"],
        "sadness": ["sadness", "grief"],
        "anger": ["anger", "annoyance"],
        "fear": ["fear"],
        "surprise": ["surprise", "realization"],
        "disgust": ["disgust"],
        "love": ["love", "caring"],
        "optimism": ["optimism"],
        "pride": ["pride"],
        "gratitude": ["gratitude"],
        "disappointment": ["disappointment"],
        "anxiety": ["nervousness"],
        "neutral": ["neutral"]
    }
    
    mapped_scores = {k: 0.0 for k in targets.keys()}
    
    for label, score in raw_results.items():
        for t_key, t_labels in targets.items():
            if label in t_labels:
                mapped_scores[t_key] += score
                break
                
    # Normalize to 1.0 (some labels might have been ignored)
    total = sum(mapped_scores.values())
    if total > 0:
        mapped_scores = {k: v / total for k, v in mapped_scores.items()}
    else:
        mapped_scores["neutral"] = 1.0
        mapped_scores = {k: v for k, v in mapped_scores.items() if k == "neutral" or v == 0.0}
        
    sorted_emotions = sorted(mapped_scores.items(), key=lambda x: x[1], reverse=True)
    
    top_emotion, top_score = sorted_emotions[0]
    
    # Rules
    if top_score < 0.40:
        top_label = "neutral"
    else:
        top_label = top_emotion
        
    # Mixed emotion check
    if len(sorted_emotions) > 1 and top_label != "neutral":
        second_emotion, second_score = sorted_emotions[1]
        if (top_score - second_score) < 0.15:
            top_label = f"Mixed Emotion: {top_emotion} + {second_emotion}"
            
    return {
        "top_emotion": top_label,
        "probabilities": dict(sorted_emotions)
    }

def process_toxicity(raw_results):
    """
    Map toxic-bert labels to: hateful, targeted, aggressive, neutral.
    """
    # Heuristic mapping
    hateful = raw_results.get("identity_hate", 0) + (raw_results.get("severe_toxic", 0) * 0.5)
    targeted = raw_results.get("threat", 0) + (raw_results.get("insult", 0) * 0.5)
    aggressive = raw_results.get("toxic", 0) + (raw_results.get("obscene", 0) * 0.5)
    
    max_tox = max(raw_results.values()) if raw_results else 0
    neutral = 1.0 - max_tox
    if neutral < 0: neutral = 0.0
    
    scores = {
        "hateful": hateful,
        "targeted": targeted,
        "aggressive": aggressive,
        "neutral": neutral
    }
    
    # Normalize
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    else:
        scores["neutral"] = 1.0
        scores = {k: v for k, v in scores.items() if k == "neutral" or v == 0.0}
        
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cat = sorted_scores[0][0]
    
    return {
        "top_category": top_cat,
        "probabilities": dict(sorted_scores)
    }