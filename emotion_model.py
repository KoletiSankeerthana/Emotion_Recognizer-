import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from utils import process_emotions, split_into_clauses

MODEL_NAME = "SamLowe/roberta-base-go_emotions"

@torch.no_grad()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

def predict_emotion(text, tokenizer, model):
    clauses = split_into_clauses(text)
    
    clause_results = []
    
    for clause in clauses:
        inputs = tokenizer(clause, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=1).squeeze()
        labels = model.config.id2label
        
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
            
        raw_results = {}
        for i, prob in enumerate(probs):
            raw_results[labels[i]] = float(prob)
            
        clause_results.append({
            "clause": clause,
            "results": process_emotions(raw_results)
        })
        
    # Full sentence prediction
    inputs_full = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs_full = model(**inputs_full)
        
    probs_full = F.softmax(outputs_full.logits, dim=1).squeeze()
    if probs_full.dim() == 0:
        probs_full = probs_full.unsqueeze(0)
        
    raw_results_full = {}
    for i, prob in enumerate(probs_full):
        raw_results_full[labels[i]] = float(prob)
        
    full_sentence_results = process_emotions(raw_results_full)
    
    return {
        "full_sentence": full_sentence_results,
        "clauses": clause_results
    }