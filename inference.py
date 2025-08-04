import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import load_model
from dataset import AIDetectionDataset
from sklearn.metrics import classification_report

@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    preds, ids = [], []
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        probs = torch.softmax(logits, dim=1)[:,1]  # P(AI)
        preds.extend(probs.cpu().numpy())
    return preds

def aggregate_predictions(df_chunks, preds, threshold=0.5):
    df_chunks["pred_score"] = preds
    df_chunks["pred_label"] = (df_chunks["pred_score"] > threshold).astype(int)
    df_grouped = df_chunks.groupby("id").agg({
        "pred_score": "mean",  # average over chunks
        "label": "first"
    })
    df_grouped["pred_label"] = (df_grouped["pred_score"] > threshold).astype(int)

    print("\n=== Document-Level Classification Report ===")
    print(classification_report(df_grouped["label"], df_grouped["pred_label"]))

    return df_grouped

def main(
    model_path="outputs/roberta_ai_detector.pt",
    model_name="roberta-base",
    input_csv="data/processed/test.csv",
    batch_size=16,
    max_length=512,
    threshold=0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("\nLoaded model")

    # Load test data
    df = pd.read_csv(input_csv)
    test_ds = AIDetectionDataset(df, tokenizer, max_length=max_length)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    print("\nLoaded test data")

    # Predict
    preds = run_inference(model, test_dl, device)
    print("\nPredictions complete")

    # Aggregate & evaluate
    doc_preds = aggregate_predictions(df, preds, threshold)
    print("\nPredictions aggregated")

    # Save predictions
    doc_preds.to_csv("data/processed/test_predictions.csv")
    print("\nSaved to `data/processed/test_predictions.csv`")

if __name__ == "__main__":
    main()
