import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def train(model, train_ds, val_ds, tokenizer, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    optimizer = AdamW(model.parameters(), lr=float(config["lr"]))
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config["epochs"] * len(train_loader)
    )

    best_f1 = 0.0
    for epoch in range(config["epochs"]):
        model.train()
        for batch in tqdm(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Validation
        preds, truths = [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
                labels = batch["label"].to(device)

                outputs = model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1)
                preds.extend(pred.cpu().numpy())
                truths.extend(labels.cpu().numpy())

        acc = accuracy_score(truths, preds)
        f1 = f1_score(truths, preds)
        print(f"[Epoch {epoch+1}] Val Accuracy: {acc:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
            torch.save(model.state_dict(), config["save_path"])
            print("🔒 Best model saved.")

# Finished loading model
# 100%|██████████████████████████████████████████████████████████████| 47443/47443 [228:54:07<00:00, 17.37s/it]
# [Epoch 1] Val Accuracy: 0.9366, F1: 0.9351 (~8 hours)