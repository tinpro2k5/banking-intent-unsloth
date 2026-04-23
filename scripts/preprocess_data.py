# scripts/preprocess_data.py
import json
import math
from pathlib import Path

import pandas as pd, yaml
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template
from utils import formatting_prompts_func

SEED = 42  # for random state in train_test_split
NUM_INTENTS = 36
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "sample_data"


PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "test": "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
}


with open(CONFIG_DIR / "preprocess.yaml") as f:
    cfg = yaml.safe_load(f)

with open(CONFIG_DIR / cfg["label_text_map"]) as f:
    label_text_map = json.load(f)
id2label = {int(k): v for k, v in label_text_map.items()}



def choose_recommended_max_len(p99_len: int) -> int:
    buckets = [16,32, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024]
    for bucket in buckets:
        if p99_len <= bucket:
            return bucket
    return 512


def compute_token_length_stats(texts: pd.Series, model_name: str, tokenizer: AutoTokenizer) -> dict:
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
        "recommended_max_length": choose_recommended_max_len(p99),
    }

def load_banking77() -> pd.DataFrame:
    # `load_dataset("PolyAI/banking77")` now fails with newer `datasets`
    # versions because that repo still has a legacy loading script on `main`.
    df_train = pd.read_parquet(PARQUET_URLS["train"])
    df_test = pd.read_parquet(PARQUET_URLS["test"])
    return pd.concat([df_train, df_test], ignore_index=True)

df_all = load_banking77()

# Select the top NUM_INTENTS most frequent intents and keep original labels.
selected_labels = (
    df_all["label"].value_counts().head(NUM_INTENTS).index.tolist()
)
df_subset = df_all[df_all["label"].isin(selected_labels)].copy()


tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
tokenizer = get_chat_template(tokenizer, chat_template=cfg["chat_template"])

df_subset_temp = df_subset.copy()
df_subset_temp["label"] = df_subset_temp["label"].map(id2label)
prompt_texts = pd.Series(formatting_prompts_func(df_subset_temp, tokenizer=tokenizer)["text"])



# Save outputs inside the project regardless of the current working directory.
CONFIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Estimate a practical max_seq_length from tokenized lengths on real data.
prompt_text_length_stats = compute_token_length_stats(
    prompt_texts,
    model_name=cfg["model_name"],
    tokenizer=tokenizer
)



label_length_stats = compute_token_length_stats(
    pd.Series(label_text_map.values()),
    model_name=cfg["model_name"],
    tokenizer=tokenizer
)

with open(CONFIG_DIR / "prompt_text_stats.json", "w", encoding="utf-8") as f:
    json.dump(prompt_text_length_stats, f, indent=2)

with open(CONFIG_DIR / "label_text_stats.json", "w", encoding="utf-8") as f:
    json.dump(label_length_stats, f, indent=2)

train_df, temp_df = train_test_split(
    df_subset,
    test_size=0.3,
    stratify=df_subset["label"],
    random_state=SEED,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=SEED,
)

train_df.to_csv(DATA_DIR / "train.csv", index=False)
val_df.to_csv(DATA_DIR / "val.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

print(
    f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}, Intents: {NUM_INTENTS}. "
    f"Saved files to {DATA_DIR} and {CONFIG_DIR}. "
    f"Recommended max_seq_length: {prompt_text_length_stats['recommended_max_length']} "
    f"(p95={prompt_text_length_stats['p95']}, p99={prompt_text_length_stats['p99']}, max={prompt_text_length_stats['max']})."
)
