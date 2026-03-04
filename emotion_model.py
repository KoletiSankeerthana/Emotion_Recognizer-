import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from utils import process_emotions

MODEL_NAME = "SamLowe/roberta-base-go_emotions"

@torch.cache
@torch.no_grad()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

def predict_emotion(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=1).squeeze()
    labels = model.config.id2label
    
    if probs.dim() == 0:
        probs = probs.unsqueeze(0)
        
    raw_results = {}
    for i, prob in enumerate(probs):
        raw_results[labels[i]] = float(prob)
        
    return process_emotions(raw_results)