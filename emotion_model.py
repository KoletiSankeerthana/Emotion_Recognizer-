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
        outputs_full = model(**inputs_full, output_attentions=True)
        
    probs_full = F.softmax(outputs_full.logits, dim=1).squeeze()
    if probs_full.dim() == 0:
        probs_full = probs_full.unsqueeze(0)
        
    raw_results_full = {}
    for i, prob in enumerate(probs_full):
        raw_results_full[labels[i]] = float(prob)
        
    full_sentence_results = process_emotions(raw_results_full)
    
    # Emotional Trigger Extraction via Attention Weights
    try:
        last_layer_attn = outputs_full.attentions[-1] # shape: (batch_size, num_heads, seq_len, seq_len)
        cls_attn = last_layer_attn[0, :, 0, :].mean(dim=0) # Mean across heads for CLS token
        
        input_ids = inputs_full['input_ids'][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        token_attn = []
        for token, attn in zip(tokens, cls_attn.tolist()):
            flag_special = token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, '<s>', '</s>']
            if not flag_special:
                clean_t = token.replace('Ġ', '').strip().strip(".,!?\"'")
                if clean_t and len(clean_t) > 2:
                    token_attn.append((clean_t, attn))
                    
        # Sort tokens by attention score
        token_attn = sorted(token_attn, key=lambda x: x[1], reverse=True)
        
        trigger_words = []
        for t, a in token_attn:
            if t.lower() not in [tw.lower() for tw in trigger_words]:
                trigger_words.append(t)
            if len(trigger_words) >= 3:
                break
    except Exception:
        trigger_words = []
        
    full_sentence_results["trigger_words"] = trigger_words
    
    return {
        "full_sentence": full_sentence_results,
        "clauses": clause_results
    }