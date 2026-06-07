# mlops-group36-iitj/src/utils.py
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def build_hf_datasets(data_dir, tokenizer, max_length=128):
    """
    Loads CSVs from data_dir, tokenizes them, and returns Hugging Face Dataset objects.
    """
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df   = pd.read_csv(f"{data_dir}/validation.csv")
    test_df  = pd.read_csv(f"{data_dir}/test.csv")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=max_length)

    cols = ["input_ids", "attention_mask", "labels"]

    def prepare(df):
        ds = Dataset.from_pandas(df)
        ds = ds.map(tokenize, batched=True)
        ds = ds.rename_column("label", "labels")
        ds.set_format("torch", columns=cols)
        return ds

    return prepare(train_df), prepare(val_df), prepare(test_df)

def compute_metrics(pred):
    """
    Standard metrics for classification tasks.
    """
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy"    : accuracy_score(labels, preds),
        "f1_weighted" : f1_score(labels, preds, average="weighted"),
        "f1_macro"    : f1_score(labels, preds, average="macro"),
        "precision"   : precision_score(labels, preds, average="weighted"),
        "recall"      : recall_score(labels, preds, average="weighted"),
    }
