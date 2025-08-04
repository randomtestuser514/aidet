import pandas as pd
import torch
from transformers import AutoTokenizer
from model import load_model
from dataset import AIDetectionDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

def predict_chunks(model, dataloader, device):
    model.eval()
    all_preds, all_ids = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            all_preds.extend(probs)
    return all_preds

def aggregate_and_score(df_chunks, preds, threshold=0.5):
    df_chunks["pred_score"] = preds
    df_chunks["pred_label"] = (df_chunks["pred_score"] > threshold).astype(int)
    df_grouped = df_chunks.groupby("id").agg({
        "pred_score": "mean",
        "label": "first"
    })
    df_grouped["pred_label"] = (df_grouped["pred_score"] > threshold).astype(int)
    print(classification_report(df_grouped["label"], df_grouped["pred_label"]))
