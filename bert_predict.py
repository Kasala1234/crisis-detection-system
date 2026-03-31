from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model
model_path = "models/bert_model"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    pred = torch.argmax(probs).item()
    confidence = torch.max(probs).item()
    
    return pred, confidence