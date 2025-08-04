import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42  # for consistency

def safe_detect_lang(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

def load_pan_data(data_dir: Path) -> List[Dict]:
    all_data = []
    for split in ["train.jsonl", "val.jsonl"]:
        path = data_dir / split
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing {split}", leave=False):
                obj = json.loads(line)
                lang = safe_detect_lang(obj["text"])
                all_data.append({
                    "id": obj["id"],
                    "text": obj["text"],
                    "label": obj["label"],
                    "source": "pan",
                    "domain": obj.get("genre", "unknown"),
                    "model": obj.get("model", None),
                    "lang": lang
                })
    return all_data

def load_raid_data(data_path: Path) -> List[Dict]:
    df = pd.read_csv(data_path)
    data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading RAID"):
        text = row["generation"]
        lang = safe_detect_lang(text)
        data.append({
            "id": row["id"],
            "text": text,
            "label": 0 if row["model"] == "human" else 1,
            "source": "raid",
            "domain": row["domain"],
            "model": row["model"],
            "lang": lang
        })
    return data

def load_m4_data(data_dir: Path) -> List[Dict]:
    all_data = []
    for file in tqdm(list(data_dir.glob("*.jsonl")), desc="Loading M4 files"):
        is_human = "_human" in file.name
        parts = file.stem.split("_")
        domain = parts[0]
        model = "human" if is_human else "_".join(parts[1:])

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if is_human:
                    text = obj.get("human_text") or obj.get("text")
                    label = 0
                else:
                    text = obj.get("machine_text") or obj.get("text")
                    label = 1

                lang = safe_detect_lang(text)

                all_data.append({
                    "id": obj.get("source_ID", "") or f"{file.name}_{hash(text)}",
                    "text": text,
                    "label": label,
                    "source": "m4",
                    "domain": domain,
                    "model": model,
                    "lang": lang
                })
    return all_data

def load_all_datasets(
    pan_path: str = "data/pan_data",
    raid_path: str = "data/raid_data/train.csv",
    m4_path: str = "data/m4_data",
    save_path: str = "data/processed/full_dataset.csv"
) -> pd.DataFrame:
    pan_data = load_pan_data(Path(pan_path))
    print("Finished loading PAN")
    raid_data = load_raid_data(Path(raid_path))
    print("Finished loading RAID")
    m4_data = load_m4_data(Path(m4_path))
    print("Finished loading M4")

    all_data = pan_data + raid_data + m4_data
    df = pd.DataFrame(all_data)

    # Filter out missing/short/invalid texts
    df = df[df["text"].notna() & (df["text"].str.len() > 20)].reset_index(drop=True)

    # Save to disk
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df
