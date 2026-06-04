import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(model_name: str, hf_token: str | None = None):
    # Load explicitly instead of using the generic pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
    
    # Set to evaluation mode (turns off dropout layers for deterministic predictions)
    model.eval() 
    return model, tokenizer

def predict(model, tokenizer, text: str) -> dict:
    # 1. Tokenize the input text into PyTorch tensors
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # 2. THE FIX: Explicitly remove token_type_ids if they exist
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # 3. Pass the clean tensors into the model
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 4. Convert raw logits into human-readable probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    # 5. Extract the winning label and score
    pred_idx = torch.argmax(probs).item()
    score = probs[pred_idx].item()
    label = model.config.id2label[pred_idx]

    return {
        "text":  text,
        "label": label,
        "score": round(score, 4),
    }

def main():
    hf_token  = os.environ.get("HF_TOKEN")
    model_name = os.environ.get("HF_MODEL", "nagaananth/MLOPS_group-v2")  
    input_text = os.environ.get("INPUT_TEXT", "Congratulations! You've won a free iPhone. Click here now.")

    if not input_text:
        print("ERROR: INPUT_TEXT environment variable is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Model  : {model_name}")
    print(f"Input  : {input_text}")
    
    model, tokenizer = load_model_and_tokenizer(model_name, hf_token)
    prediction = predict(model, tokenizer, input_text)
    
    print("\n=== Prediction Result ===")
    print(json.dumps(prediction, indent=2))

if __name__ == "__main__":
    main()
