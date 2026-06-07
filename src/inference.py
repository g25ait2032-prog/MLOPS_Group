import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login, whoami

# ---------------------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name: str, hf_token: str | None = None):
    print(f"Loading model: {model_name} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
    
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def predict(model, tokenizer, text: str) -> dict:
    # Use 'with torch.inference_mode()' for better performance than 'torch.no_grad()'
    with torch.inference_mode():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True, # Added padding for robustness
            max_length=128
        ).to(DEVICE)
        
        # Modern models often don't need token_type_ids; 
        # model(**inputs) automatically ignores keys that aren't in the model's forward() signature.
        outputs = model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()
        predicted_label = model.config.id2label[predicted_index]

    return {
        "text": text,
        "label": predicted_label,
        "confidence": round(confidence, 4)
    }

# ---------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------
def main():

    hf_token = os.getenv("HF_TOKEN")

    model_name = os.getenv(
        "HF_MODEL",
        "nagaananth/MLOPS_group-v1"
    )

    input_text = os.getenv(
        "INPUT_TEXT",
        "Congratulations! You've won a free iPhone. Click here now."
    )

    if not input_text.strip():
        print(
            "ERROR: INPUT_TEXT is empty.",
            file=sys.stderr
        )
        sys.exit(1)

    # -------------------------------------------------------------
    # Hugging Face Authentication
    # -------------------------------------------------------------
    if hf_token:
        try:
            login(token=hf_token)

            user_info = whoami()

            print(
                f"Authenticated as: {user_info['name']}"
            )

        except Exception as error:
            print(
                f"Hugging Face login failed: {error}"
            )

    # -------------------------------------------------------------
    # Load Model
    # -------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        hf_token=hf_token
    )

    # -------------------------------------------------------------
    # Run Prediction
    # -------------------------------------------------------------
    result = predict(
        model=model,
        tokenizer=tokenizer,
        text=input_text
    )

    # -------------------------------------------------------------
    # Display Output
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Prediction Result")
    print("=" * 50)

    print(
        json.dumps(
            result,
            indent=2
        )
    )


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
