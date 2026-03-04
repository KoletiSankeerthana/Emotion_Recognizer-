import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import process_toxicity

MODEL_NAME = "unitary/toxic-bert"

@torch.cache
@torch.no_grad()
def load_hate_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

def predict_toxicity(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.sigmoid(outputs.logits).squeeze()
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    if probs.dim() == 0:
        probs = probs.unsqueeze(0)
        
    raw_results = {}
    for i, label in enumerate(labels):
        raw_results[label] = float(probs[i])
        
    return process_toxicity(raw_results)
