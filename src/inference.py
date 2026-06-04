import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login, whoami


def load_model_and_tokenizer(model_name: str, hf_token: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        token=hf_token
    )

    model.eval()
    return model, tokenizer


def predict(model, tokenizer, text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    pred_idx = torch.argmax(probs).item()
    score = probs[pred_idx].item()
    label = model.config.id2label[pred_idx]

    return {
        "text": text,
        "label": label,
        "score": round(score, 4)
    }


def main():
    hf_token = os.getenv("HF_TOKEN")
    model_name = os.getenv(
        "HF_MODEL",
        "nagaananth/MLOPS_group-v2"
    )
    input_text = os.getenv(
        "INPUT_TEXT",
        "Congratulations! You've won a free iPhone. Click here now."
    )

    if not input_text:
        print(
            "ERROR: INPUT_TEXT environment variable is empty.",
            file=sys.stderr
        )
        sys.exit(1)

    if hf_token:
        try:
            login(token=hf_token)
            print(
                f"Authenticated as: {whoami()['name']}"
            )
        except Exception as e:
            print(
                f"HF authentication failed: {e}"
            )

    print(f"Model : {model_name}")
    print(f"Input : {input_text}")

    model, tokenizer = load_model_and_tokenizer(
        model_name,
        hf_token
    )

    prediction = predict(
        model,
        tokenizer,
        input_text
    )

    print("\n=== Prediction Result ===")
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
