import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

@torch.no_grad()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

def predict_emotion(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits.squeeze()
    probs = F.softmax(outputs.logits, dim=1).squeeze()
    
    labels = model.config.id2label
    
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
        probs = probs.unsqueeze(0)
        
    results = {}
    raw_logits = {}
    
    for i, prob in enumerate(probs):
        results[labels[i]] = float(prob)
        raw_logits[labels[i]] = float(logits[i])
        
    # Sort by probability
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    emotions = list(sorted_results.items())
    top_emotion, top_score = emotions[0]
    
    is_neutral_override = False
    if top_score < 0.40:
        top_emotion = "neutral"
        is_neutral_override = True
        
    mixed_emotions = []
    if len(emotions) > 1 and not is_neutral_override:
        second_emotion, second_score = emotions[1]
        if (top_score - second_score) < 0.15:
            mixed_emotions = [top_emotion, second_emotion]
            
    return {
        "top_emotion": top_emotion,
        "confidence": float(top_score),
        "mixed_emotions": mixed_emotions,
        "is_neutral_override": is_neutral_override,
        "probabilities": sorted_results,
        "logits": raw_logits
    }