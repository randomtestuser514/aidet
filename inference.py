import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from dataset import AIDetectionDataset
from model import load_model
from pathlib import Path
import argparse
import numpy as np

@torch.no_grad()
def predict_chunks(model, dataloader, device):
    model.eval()
    all_probs = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(AI)
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)

def aggregate_predictions(df, chunk_preds, threshold=0.5):
    df = df.copy()
    df["pred_score"] = chunk_preds

    def safe_mode(series):
        return series.mode().iloc[0] if not series.mode().empty else "unknown"

    grouped = df.groupby("id").agg({
        "pred_score": "mean",
        "label": "first",
        "model": safe_mode,
        "lang": safe_mode,
        "text": lambda texts: " ".join(texts)
    }).reset_index()

    grouped["pred_label"] = (grouped["pred_score"] > threshold).astype(int)
    grouped["word_count"] = grouped["text"].str.split().apply(len)

    # Debug: How many missing?
    missing_langs = grouped["lang"].isna().sum()
    missing_models = grouped["model"].isna().sum()
    if missing_langs > 0 or missing_models > 0:
        print(f"⚠️ Warning: {missing_langs} entries missing lang, {missing_models} missing model")

    return grouped

def evaluate_slices(df, label_col="label", pred_col="pred_label", show_small_groups=True):
    def print_grouped_metrics(groupby_col, title):
        print(f"\n📊 Breakdown by {title}:\n" + "-" * 40)
        for val, subset in df.groupby(groupby_col):
            acc = accuracy_score(subset[label_col], subset[pred_col])
            f1 = f1_score(subset[label_col], subset[pred_col])
            try:
                auc = roc_auc_score(subset[label_col], subset["pred_score"])
            except:
                auc = np.nan

            size_flag = "" if len(subset) >= 10 else "(few)"
            print(f"{str(val):<15} | Samples: {len(subset):>4} {size_flag:<6} | Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    print_grouped_metrics("lang", "Language")
    print_grouped_metrics("model", "Model Source")

    df["length_bin"] = pd.cut(df["word_count"], bins=[0, 50, 100, 200, 300, 500, 1000, 10000])
    print_grouped_metrics("length_bin", "Length")

def main(
    model_path="outputs/roberta_ai_detector.pt",
    model_name="roberta-base",
    input_csv="data/processed/test.csv",
    output_csv="data/processed/test_predictions.csv",
    batch_size=16,
    max_length=512,
    threshold=0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load chunked test data
    df = pd.read_csv(input_csv)
    ds = AIDetectionDataset(df, tokenizer, max_length=max_length)
    dl = DataLoader(ds, batch_size=batch_size)

    print(f"\n\nRunning inference on {len(df)} chunked samples...")
    chunk_preds = predict_chunks(model, dl, device)

    print("\nAggregating document-level predictions...")
    doc_df = aggregate_predictions(df, chunk_preds, threshold=threshold)

    print("\n\nOverall Evaluation:\n" + "-" * 40)
    acc = accuracy_score(doc_df["label"], doc_df["pred_label"])
    f1 = f1_score(doc_df["label"], doc_df["pred_label"])
    auc = roc_auc_score(doc_df["label"], doc_df["pred_score"])
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print(f"✅ AUC Score: {auc:.4f}")

    evaluate_slices(doc_df)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    doc_df.to_csv(output_csv, index=False)
    print(f"\n\nSaved predictions to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/roberta_ai_detector.pt")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--input_csv", type=str, default="data/processed/test.csv")
    parser.add_argument("--output_csv", type=str, default="data/processed/test_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(**vars(args))
