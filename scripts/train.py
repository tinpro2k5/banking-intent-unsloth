# scripts/train.py
import yaml, json, pandas as pd, torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Load config
with open("configs/train.yaml") as f:
    cfg = yaml.safe_load(f)

with open(cfg["label_map"]) as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)

# Load model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["model_name"],
    max_seq_length=cfg["max_seq_length"],
    dtype=None,
    load_in_4bit=True,
)


# max_seq_length = 2048 – Controls context length. While Llama-3 supports 8192, we recommend 2048 for testing. Unsloth enables 4× longer context fine-tuning.

# dtype = None – Defaults to None; use torch.float16 or torch.bfloat16 for newer GPUs.

# load_in_4bit = True – Enables 4-bit quantization, reducing memory use 4× for fine-tuning. Disabling it enables LoRA 16-bit fine-tuning. You can also enable 16-bit LoRA with load_in_16bit = True

# To enable full fine-tuning (FFT), set full_finetuning = True. For 8-bit fine-tuning, set load_in_8bit = True.

# Note: Only one training method can be set to True at a time.

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=cfg["lora_r"],
    target_modules=["q_proj", "v_proj"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    bias="none",
    use_gradient_checkpointing=True,
)

# Prepare data — format as instruction-style prompt
def format_row(row):
    label_name = id2label[row["label"]]
    return {
        "text": f"Classify the banking intent of this message.\nMessage: {row['text']}\nIntent: {label_name}"
    }

train_df = pd.read_csv(cfg["train_data"])
test_df  = pd.read_csv(cfg["test_data"])

train_dataset = Dataset.from_pandas(train_df).map(format_row)
test_dataset  = Dataset.from_pandas(test_df).map(format_row)

# Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=cfg["max_seq_length"],
    args=TrainingArguments(
        per_device_train_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        optim=cfg["optimizer"],
        output_dir=cfg["output_dir"],
        save_strategy="epoch",
        logging_steps=10,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    ),
)

trainer.train()
model.save_pretrained(cfg["output_dir"])
tokenizer.save_pretrained(cfg["output_dir"])
print("✅ Training done. Checkpoint saved to:", cfg["output_dir"])