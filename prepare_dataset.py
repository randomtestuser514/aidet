import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import random

def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    label_counts = df["label"].value_counts()
    min_count = label_counts.min()

    print(f"Balancing classes to {min_count} samples each...")
    dfs = []
    for label in [0, 1]:
        class_subset = df[df["label"] == label]
        dfs.append(class_subset.sample(n=min_count, random_state=42))

    balanced_df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

def chunk_text(text, max_words=300):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def prepare_for_training(
    input_path: str = "data/processed/full_dataset.csv",
    output_dir: str = "data/processed",
    keep_lang: str = None,
    max_words: int = 300,
    do_chunking: bool = True,
    balance: bool = True,
    val_size: float = 0.1,
    test_size: float = 0.1,
):
    df = pd.read_csv(input_path)

    print(f"Initial size: {len(df)}")

    # Filter by language
    if keep_lang:
        df = df[df["lang"] == keep_lang]
        print(f"After language filtering: {len(df)}")

    # Balance dataset
    if balance:
        df = balance_classes(df)
        print(f"After balancing: {len(df)}")

    # Chunk long texts
    if do_chunking:
        print("Chunking long texts...")
        chunked_rows = []
        for idx, row in df.iterrows():
            chunks = chunk_text(row["text"], max_words=max_words)
            for i, chunk in enumerate(chunks):
                chunked_rows.append({
                    "id": f"{row['id']}_chunk{i}",
                    "text": chunk,
                    "label": row["label"],
                    "source": row["source"],
                    "domain": row["domain"],
                    "model": row["model"],
                    "lang": row["lang"]
                })
        df = pd.DataFrame(chunked_rows)
        print(f"After chunking: {len(df)}")

    # Stratified Train/Val/Test split
    print("Splitting into train/val/test...")
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size / (1 - test_size), stratify=train_val_df["label"], random_state=42)

    # Save splits
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(Path(output_dir) / "train.csv", index=False)
    val_df.to_csv(Path(output_dir) / "val.csv", index=False)
    test_df.to_csv(Path(output_dir) / "test.csv", index=False)

    print(f"Final dataset sizes:\n  Train: {len(train_df)}\n  Val: {len(val_df)}\n  Test: {len(test_df)}")

if __name__ == "__main__":
    prepare_for_training()
    df = pd.read_csv("data/processed/train.csv")
    print("\n\n==TRAIN==")
    # print(df['lang'].value_counts())
    print(df['label'].value_counts())

    df = pd.read_csv("data/processed/val.csv")
    print("\n\n==VAL==")
    # print(df['lang'].value_counts())
    print(df['label'].value_counts())

    df = pd.read_csv("data/processed/test.csv")
    print("\n\n==TEST==")
    # print(df['lang'].value_counts())
    print(df['label'].value_counts())
