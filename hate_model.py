import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_NAME = "unitary/toxic-bert"

@torch.no_grad()
def load_hate_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

def predict_toxicity(text, tokenizer, model):
    # Truncate to maximum length accepted by BERT (512 tokens max)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use sigmoid since toxic-bert is trained with Binary Cross Entropy for multilabel
    probs = torch.sigmoid(outputs.logits).squeeze()
    
    # unitary/toxic-bert labels according to literature
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    if probs.dim() == 0:
        probs = probs.unsqueeze(0)
        
    results = {}
    for i, label in enumerate(labels):
        results[label] = float(probs[i])
        
    # Determine overall toxicity score
    toxicity_score = max(results.values())
    
    # Risk level classification
    if toxicity_score < 0.30:
        risk_level = "Low"
    elif toxicity_score <= 0.70:
        risk_level = "Medium"
    else:
        risk_level = "High"
        
    return {
        "toxicity_score": toxicity_score,
        "risk_level": risk_level,
        "probabilities": results
    }
