import os
import sys
import json
from transformers import pipeline
from huggingface_hub import login

def load_pipeline(model_name: str, hf_token: str | None = None):
    if hf_token:
        login(token=hf_token)
    clf = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=-1,           # CPU inference for container
    )
    return clf

def predict(clf, text: str) -> dict:
    result = clf(text, truncation=True, max_length=128)[0]
    return {
        "text":  text,
        "label": result["label"],
        "score": round(result["score"], 4),
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
    
    clf = load_pipeline(model_name, hf_token)
    prediction = predict(clf, input_text)
    
    print("\n=== Prediction Result ===")
    print(json.dumps(prediction, indent=2))

if __name__ == "__main__":
    main()
