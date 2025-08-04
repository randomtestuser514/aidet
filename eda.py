import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")

def run_eda(dataset_path: str = "data/processed/full_dataset.csv", output_dir: str = "eda_outputs"):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # === 1. Label Distribution ===
    label_counts = df['label'].value_counts().sort_index()
    print("\nLabel Distribution (0=Human, 1=AI):")
    print(label_counts)

    plt.figure(figsize=(5,4))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="Set2")
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks([0, 1], ["Human", "AI"])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/label_distribution.png")
    plt.close()

    # === 2. Language Distribution ===
    lang_counts = df['lang'].value_counts()
    print("\nTop 10 Languages:")
    print(lang_counts.head(10))

    plt.figure(figsize=(8,5))
    sns.barplot(x=lang_counts.head(10).index, y=lang_counts.head(10).values, palette="coolwarm")
    plt.title("Top 10 Languages")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/language_distribution.png")
    plt.close()

    # === 3. Domain Distribution ===
    domain_counts = df['domain'].value_counts()
    print("\nTop 10 Domains:")
    print(domain_counts.head(10))

    plt.figure(figsize=(8,5))
    sns.barplot(x=domain_counts.head(10).index, y=domain_counts.head(10).values, palette="muted")
    plt.title("Top 10 Domains")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/domain_distribution.png")
    plt.close()

    # === 4. Model Usage (AI Only) ===
    ai_df = df[df["label"] == 1]
    model_counts = ai_df['model'].value_counts()
    print("\nTop 10 AI Models:")
    print(model_counts.head(10))

    plt.figure(figsize=(8,5))
    sns.barplot(x=model_counts.head(10).index, y=model_counts.head(10).values, palette="Set3")
    plt.title("Top 10 AI Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_distribution.png")
    plt.close()

    # === 5. Text Length Distribution ===
    df['char_length'] = df['text'].str.len()
    df['word_length'] = df['text'].str.split().apply(len)

    print("\nText Length Stats:")
    print(df[['char_length', 'word_length']].describe())

    plt.figure(figsize=(8,5))
    sns.histplot(df['word_length'], bins=50, kde=True)
    plt.title("Text Length Distribution (Words)")
    plt.xlabel("Number of Words")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/word_length_distribution.png")
    plt.close()

    print(f"\nEDA complete. Visualizations saved to `{output_dir}/`.")

if __name__ == "__main__":
    run_eda()
