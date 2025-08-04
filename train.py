import pandas as pd
import yaml
from transformers import AutoTokenizer
from dataset import AIDetectionDataset
from model import load_model
from trainer import train

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    df_train = pd.read_csv("data/processed/train.csv")
    df_val = pd.read_csv("data/processed/val.csv")
    print("Finished loading in data")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_ds = AIDetectionDataset(df_train, tokenizer, max_length=config["max_length"])
    val_ds = AIDetectionDataset(df_val, tokenizer, max_length=config["max_length"])
    print("Finished setting up datasets")

    model = load_model(model_name=config["model_name"])
    print("Finished loading model")
    train(model, train_ds, val_ds, tokenizer, config)
