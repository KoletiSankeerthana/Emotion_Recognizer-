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
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).squeeze()

    labels = model.config.id2label
    results = {}

    for i, prob in enumerate(probs):
        results[labels[i]] = float(prob)

    # Sort by probability
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    return sorted_results