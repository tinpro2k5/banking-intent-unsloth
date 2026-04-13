# scripts/preprocess_data.py
import json
import math
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

SEED = 42  # for random state in train_test_split
NUM_INTENTS = 20
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "sample_data"

PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "test": "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
}


def choose_recommended_max_len(p99_len: int) -> int:
    buckets = [64, 96, 128, 160, 192, 256, 384, 512]
    for bucket in buckets:
        if p99_len <= bucket:
            return bucket
    return 512


def compute_token_length_stats(texts: pd.Series, model_name: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encodings = tokenizer(
        texts.astype(str).tolist(),
        add_special_tokens=True,
        truncation=False,
    )
    lengths = pd.Series([len(ids) for ids in encodings["input_ids"]], dtype="int64")

    p50 = int(math.ceil(lengths.quantile(0.50)))
    p95 = int(math.ceil(lengths.quantile(0.95)))
    p99 = int(math.ceil(lengths.quantile(0.99)))
    max_len = int(lengths.max())

    return {
        "model_name": model_name,
        "num_samples": int(len(lengths)),
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "max": max_len,
        "recommended_max_seq_length": choose_recommended_max_len(p99),
    }


def load_banking77() -> pd.DataFrame:
    # `load_dataset("PolyAI/banking77")` now fails with newer `datasets`
    # versions because that repo still has a legacy loading script on `main`.
    dataset = load_dataset("parquet", data_files=PARQUET_URLS)
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()
    return pd.concat([df_train, df_test], ignore_index=True)


df_all = load_banking77()

# Select the top NUM_INTENTS most frequent classes
selected_labels = df_all["label"].value_counts().head(NUM_INTENTS).index.tolist()
df_subset = df_all[df_all["label"].isin(selected_labels)].copy()

# Remap labels to 0..N-1
label_list = [int(label) for label in sorted(df_subset["label"].unique())]
label2id = {label: idx for idx, label in enumerate(label_list)}
df_subset["label"] = df_subset["label"].map(label2id)

# Save outputs inside the project regardless of the current working directory.
CONFIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

with open(CONFIG_DIR / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(label2id, f, indent=2)

# Estimate a practical max_seq_length from tokenized lengths on real data.
length_stats = compute_token_length_stats(
    df_subset["text"],
    model_name="unsloth/llama-3-8b-bnb-4bit",
)
with open(CONFIG_DIR / "data_stats.json", "w", encoding="utf-8") as f:
    json.dump(length_stats, f, indent=2)

train_df, test_df = train_test_split(
    df_subset,
    test_size=0.2,
    stratify=df_subset["label"],
    random_state=SEED,
)
train_df.to_csv(DATA_DIR / "train.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

print(
    f"Train: {len(train_df)}, Test: {len(test_df)}, Intents: {NUM_INTENTS}. "
    f"Saved files to {DATA_DIR} and {CONFIG_DIR}. "
    f"Recommended max_seq_length: {length_stats['recommended_max_seq_length']} "
    f"(p95={length_stats['p95']}, p99={length_stats['p99']}, max={length_stats['max']})."
)
