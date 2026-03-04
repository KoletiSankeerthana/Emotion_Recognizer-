import re

def split_into_clauses(text):
    # Split on conjunctions: but, however, although, yet, while
    pattern = r'\b(but|however|although|yet|while)\b'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    if len(parts) == 1:
        return [text]
        
    clauses = []
    first_part = parts[0].strip()
    if first_part:
        if first_part.endswith(','):
            first_part = first_part[:-1].strip()
        clauses.append(first_part)
        
    for i in range(1, len(parts), 2):
        next_part = parts[i+1].strip()
        if next_part:
            clauses.append(next_part)
            
    return clauses if clauses else [text]

import math

def get_entropy(probabilities):
    entropy = 0.0
    for prob in probabilities.values():
        if prob > 0:
            entropy -= prob * math.log(prob)
    return entropy

def get_sentiment(probabilities):
    pos_emotions = ["joy", "love", "optimism", "pride", "gratitude", "excitement"]
    neg_emotions = ["sadness", "anger", "fear", "disgust", "disappointment", "anxiety"]
    
    pos_score = sum(probabilities.get(e, 0) for e in pos_emotions)
    neg_score = sum(probabilities.get(e, 0) for e in neg_emotions)
    neu_score = probabilities.get("neutral", 0.0) + probabilities.get("surprise", 0.0)
    
    if pos_score > neg_score and pos_score > neu_score:
        return "Positive", pos_score, neg_score
    elif neg_score > pos_score and neg_score > neu_score:
        return "Negative", pos_score, neg_score
    else:
        return "Neutral", pos_score, neg_score

def get_suggestion(top_emotion_name):
    suggestions = {
        "fear": "You may be experiencing uncertainty. Consider preparing step-by-step to reduce anxiety.",
        "anxiety": "You might be feeling overwhelmed. Try to ground yourself and focus on one thing at a time.",
        "anger": "You appear frustrated. It might be helpful to take a pause and reflect before proceeding.",
        "sadness": "Experiencing this is difficult; consider taking some time for self-care and rest.",
        "disappointment": "It is okay to feel let down. Acknowledge the feeling and look for small positive steps forward.",
        "disgust": "You seem to be reacting strongly to something adverse. Stepping back may provide a better perspective.",
        "joy": "This is a great moment! Encourage yourself to continue and build on this positive momentum.",
        "excitement": "Harness this energy to push forward on your goals.",
        "pride": "You've likely achieved something meaningful. Take a moment to appreciate your effort.",
        "optimism": "A positive outlook is powerful. Keep focusing on the potential for good outcomes.",
        "gratitude": "Appreciating the good things fosters resilience. Hold onto this positive perspective.",
        "love": "This is a strong feeling of connection. Nurturing it can bring comfort and joy.",
        "surprise": "Unexpected events can be jarring. Take a moment to process the new information.",
        "neutral": "You seem to be in a balanced and even state of mind."
    }
    return suggestions.get(top_emotion_name, "Take a moment to process your feelings and proceed mindfully.")

def get_reliability(top_score):
    if top_score > 0.8:
        return "High confidence prediction"
    elif top_score >= 0.5:
        return "Moderate confidence prediction"
    else:
        return "Low confidence — emotional ambiguity detected"

def get_dominance_metrics(sorted_emotions):
    if len(sorted_emotions) < 2:
        return 1.0, 0.0, "Emotion strongly dominant"
        
    top_score = sorted_emotions[0][1]
    second_score = sorted_emotions[1][1]
    
    dominance = top_score - second_score
    balance = 1.0 - dominance
    
    if dominance > 0.5:
        text = "Emotion strongly dominant"
    elif dominance >= 0.2:
        text = "Moderately dominant"
    else:
        text = "Emotionally balanced / mixed"
        
    return dominance, balance, text

def process_emotions(raw_results):
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
        "excitement": ["excitement"],
        "neutral": ["neutral"]
    }
    
    mapped_scores = {k: 0.0 for k in targets.keys()}
    
    for label, score in raw_results.items():
        for t_key, t_labels in targets.items():
            if label in t_labels:
                mapped_scores[t_key] += score
                break
                
    total = sum(mapped_scores.values())
    if total > 0:
        mapped_scores = {k: v / total for k, v in mapped_scores.items()}
    else:
        mapped_scores["neutral"] = 1.0
        mapped_scores = {k: v for k, v in mapped_scores.items() if k == "neutral" or v == 0.0}
        
    sorted_emotions = sorted(mapped_scores.items(), key=lambda x: x[1], reverse=True)
    top_emotion, top_score = sorted_emotions[0]
    
    if all(score < 0.25 for name, score in sorted_emotions if name != "neutral"):
        top_label = "neutral"
    else:
        top_label = top_emotion
        
    primary_emo = f"{top_emotion} ({top_score*100:.0f}%)"
    secondary_emo = None
    
    if len(sorted_emotions) > 1 and top_label != "neutral":
        second_emotion, second_score = sorted_emotions[1]
        
        if second_score >= (0.60 * top_score):
            top_label = f"Mixed Emotion"
            primary_emo = f"{top_emotion} ({top_score*100:.0f}%)"
            secondary_emo = f"{second_emotion} ({second_score*100:.0f}%)"
            
    probabilities = dict(sorted_emotions)
    sentiment, pos_score, neg_score = get_sentiment(probabilities)
    suggestion = get_suggestion(top_emotion)
    reliability = get_reliability(top_score)
    dominance, balance, dom_text = get_dominance_metrics(sorted_emotions)
    entropy_score = get_entropy(probabilities)
            
    return {
        "top_emotion": top_label,
        "primary": primary_emo,
        "secondary": secondary_emo,
        "probabilities": probabilities,
        "sentiment": sentiment,
        "pos_score": pos_score,
        "neg_score": neg_score,
        "suggestion": suggestion,
        "reliability": reliability,
        "dominance": dominance,
        "balance": balance,
        "dominance_text": dom_text,
        "entropy": entropy_score
    }

def process_toxicity(raw_results):
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