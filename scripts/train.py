import yaml, json, pandas as pd, torch
import shutil
from datetime import datetime
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template
from pathlib import Path
from utils import formatting_prompts_func

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "sample_data"


# Load config
with open(CONFIG_DIR / "train.yaml") as f:
    cfg = yaml.safe_load(f)

# Load label to text mapping
with open(CONFIG_DIR / cfg["label_text_map"]) as f:
    label_text_map = json.load(f)
id2label = {int(k): v for k, v in label_text_map.items()}

# Load model & tokenizer with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["model_name"],
    max_seq_length=cfg["max_seq_length"],
    dtype=cfg["dtype"],
    load_in_4bit=cfg["load_in_4bit"],
)



# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=cfg["lora_r"],
    target_modules=cfg["target_modules"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    bias=cfg["bias"],
    use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
    use_rslora=cfg["use_rslora"],
    random_state=cfg["random_state"],
)

tokenizer = get_chat_template(tokenizer, chat_template=cfg["chat_template"])
# Format prompts for training

# Load and format datasets
train_df = pd.read_csv(DATA_DIR / cfg["train_data"] )
val_df = pd.read_csv(DATA_DIR / cfg["val_data"])
test_df  = pd.read_csv(DATA_DIR / cfg["test_data"] )


train_df["label"] = train_df["label"].map(id2label)
val_df["label"] = val_df["label"].map(id2label)
test_df["label"] = test_df["label"].map(id2label)

train_dataset = Dataset.from_pandas(pd.DataFrame(formatting_prompts_func(train_df, tokenizer=tokenizer)))
test_dataset = Dataset.from_pandas(pd.DataFrame(formatting_prompts_func(test_df, tokenizer=tokenizer)))
val_dataset = Dataset.from_pandas(pd.DataFrame(formatting_prompts_func(val_df, tokenizer=tokenizer)))



print("✅ Datasets loaded and formatted.") 
print("Label to text mapping:")
for label_id, label_text in id2label.items():
    print(f"{label_id}: {label_text}")
print("-------------------------------")
print("Example formatted prompt:")
for i in range(3):
    print(train_dataset[i]["text"])
    print("-------------------------------")
###################################################################



    
output_dir = (REPO_ROOT / cfg["output_dir"]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

# Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=cfg["max_seq_length"],
    args=TrainingArguments(
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        optim=cfg["optimizer"],
        output_dir=str(output_dir),
        save_strategy=cfg["save_strategy"],
        evaluation_strategy=cfg["evaluation_strategy"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        logging_steps=cfg["logging_steps"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_steps"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    ),
)


from unsloth.chat_templates import train_on_responses_only
# This training strategy focuses the model updates on the assistant's responses
# which is ideal for classification tasks where the response (label) is the main learning signal.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)


train_result = trainer.train()
model.save_pretrained(str(output_dir)) # only adapter
tokenizer.save_pretrained(str(output_dir))
trainer.save_state()

run_info = {
    "timestamp": datetime.now().isoformat(),
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
    "test_samples": len(test_dataset),
    "metrics": train_result.metrics,
    "config": cfg,
}
with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
    json.dump(run_info, f, indent=2, ensure_ascii=False)

shutil.copy2(CONFIG_DIR / "train.yaml", output_dir / "train.yaml")
shutil.copy2(CONFIG_DIR / cfg["label_text_map"], output_dir / "label_text_map.json")

print("✅ Training done. Checkpoint and metadata saved to:", output_dir)
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving