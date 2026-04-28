# Banking Intent Classification with Unsloth

## Setup
```bash
# 1. Install PyTorch with CUDA (must do first)
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
# 2. Install other dependencies
pip install -r requirements.txt
```

    Optional – Install memory-efficient attention (optional)
```bash
pip install xformers==0.0.27 --no-deps
```

Notes:
- Do NOT include torch or xformers in requirements.txt to avoid version conflicts.

## Project Structure (Guideline)

```text
banking-intent-unsloth/
├── checkpoints/               # Directory containing saved ~95MB LoRA adapters
│   └── 20260424_122036/       # Example timestamped run directory
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── train.yaml         # A copy of the configuration used for this run
├── configs/                   # Configuration files
│   ├── inference.yaml         # Inference parameters
│   ├── label_text_map.json    # Original full label dictionary (77 intents)
│   ├── train.yaml             # Training parameters (Batch size, LR, LoRA, etc.)
│   └── used_label_text_map.json # Subset of mapped intent labels used for fine-tuning
├── sample_data/               # Auto-generated dataset splits
│   ├── test.csv
│   ├── train.csv
│   └── val.csv
├── scripts/                   # Core Python logic
│   ├── evaluate.py            # Compares fine-tuned model vs base model
│   ├── inference.py           # IntentClassification class for predicting new text
│   ├── preprocess_data.py     # Downloads dataset, formats, and creates CSV splits
│   ├── train.py               # Unsloth fine-tuning loop (SFTTrainer)
│   └── utils.py               # Helper functions (e.g., ChatML prompt formatting)
├── unsloth_compiled_cache/    # Unsloth's native compilation cache for x2 speedups
├── evaluate.sh                # Bash wrapper for quick evaluation
├── inference.sh               # Bash wrapper for interactive inference loop
├── train.sh                   # Bash wrapper for kicking off the training run
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md                  # This project documentation
```

## Prepare Data (Download & Preprocess)
```bash
python scripts/preprocess_data.py
```
This script will automatically **download** the `PolyAI/banking77` dataset from Hugging Face, select the top 36 intents, clean the text, and split it into `train.csv`, `val.csv`, and `test.csv` inside the `sample_data/` directory.

## Lightweight Model & Adapter Architecture

This project is highly optimized for consumer hardware. By using a **3B parameter model in 4-bit quantization** combined with LoRA (`r=8/16`) the training footprint is incredibly small:

- **Trainable Parameters:** Only `24,313,856` out of `3,237,063,680` (approx. **0.75%** of the model is trained) for Llama 3.2 3B Instruct set up
- **Storage Efficiency:** The resulting saved checkpoint (adapter) is extremely lightweight—only about **95 MB**. 
- **GitHub Friendly:** Because the adapter is under 100 MB, it can be pushed directly to GitHub inside the `checkpoints/` directory.
- **Automatic Base Model Resolution:** When you run inference or evaluation for the first time, you do not need to manually download massive model weights. Unsloth automatically recognizes the base model reference from the adapter's config, downloads the base 3B model from Hugging Face on the fly, and seamlessly injects the 95MB adapter weights into it.

## Train
```bash
bash train.sh
```

### Known Environment Issues

This training setup is currently most reliable on a single modern GPU.

- **Single GPU is the recommended path.** The current codebase uses `Unsloth + TRL SFTTrainer + train_on_responses_only` in a normal single-process launch. This works reliably on environments such as Colab with `1x T4`.
- **Kaggle multi-GPU can fail even when single-GPU works.** On Kaggle, environments such as `2x T4` may enter a different Trainer execution path than `1x T4`, even when the code is unchanged. In practice, this can trigger failures during `trainer.train()` such as `'int' object has no attribute 'mean'`.
- **Version pinning alone may not fix Kaggle multi-GPU issues.** Even with pinned `unsloth`, `transformers`, `trl`, and `peft` versions, Kaggle multi-GPU can still behave differently from local or Colab single-GPU runs because the runtime path is different.
- **Kaggle P100 is not supported by newer PyTorch CUDA builds.** If Kaggle assigns a `Tesla P100`, newer stacks such as `torch 2.10.0 + cu128` can fail before training starts with errors like `CUDA error: no kernel image is available for execution on the device` or warnings that `sm_60` is not supported by the installed PyTorch build.

Current recommendation:

- Prefer single-GPU runs for this repository unless you are prepared to debug distributed training separately.
- If Kaggle exposes multiple GPUs, treat multi-GPU behavior as experimental for this project.
- Use a proper distributed launcher such as `accelerate launch` or `torchrun` instead of assuming that a normal `python scripts/train.py` run on a machine with multiple visible GPUs is equivalent.



## Evaluate
The evaluation script compares the fine-tuned checkpoint against the base model on the validation split and reports both accuracies.

How it works:
- By default, loads the configuration from `configs/train.yaml` (repository-level CONFIG_DIR).
- `--run-dir` specifies which checkpoint to evaluate but **does NOT automatically change the config file**.
- To evaluate a specific run with its original config (matching `base_model`, `max_seq_length`, etc.), pass `--config <run>/train.yaml`.
- Resolves a model checkpoint automatically from the latest timestamped run under `checkpoints/`, or from the specified `--run-dir`.
- Applies the same chat template used during training.
- Uses `sample_data/val.csv` by default, but you can override the split with `--split`.
- Generates one intent label for the fine-tuned checkpoint and one for the base model, then compares both against ground-truth labels and prints the accuracy gap.

Useful modes:
- `--split val|test|train` — selects which dataset split to evaluate (default: `val`).
- `--run-dir <path>` — evaluates a specific trained run under `checkpoints/` and is the normal way to score a checkpoint.
- `--config <path>` — uses a specific config file (recommended when evaluating a particular run to ensure base model matches).
- `--best-from-run` — uses the best checkpoint recorded in `trainer_state.json` within the run.

Examples:
```bash
# Default: evaluate latest run on validation split (uses configs/train.yaml)
bash evaluate.sh

# Evaluate latest run on test split (uses configs/train.yaml)
bash evaluate.sh --split test

# Evaluate a specific run with its own config (ensures correct base_model pairing)
bash evaluate.sh --run-dir checkpoints/20260424_122036 --config checkpoints/20260424_122036/train.yaml --split test
```
## Inference

You can run the interactive inference loop via the terminal:
```bash
bash inference.sh
```

### Usage Example
As required, here is a short example demonstrating how to instantiate the `IntentClassification` class and predict the intent label for a single text input in Python:

```python
from scripts.inference import IntentClassification

# 1. Initialize the inference class with the path to the configuration file
# The configuration file contains the adapter_path to the saved checkpoint
classifier = IntentClassification(model_path="configs/inference.yaml")

# 2. Predict the intent for a single text input
message = "I lost my credit card, please help me block it."
predicted_label = classifier(message)

print(f"Input: {message}")
print(f"Predicted Intent: {predicted_label}")
```

### How `inference.py` Works Under the Hood
To meet the consistency requirements, the inference process is encapsulated in the `IntentClassification` class with two main methods:

1. **`__init__(self, model_path)` (Initialization):**
   - **Configuration Loading:** Takes the path to a YAML config file (e.g., `configs/inference.yaml`) which specifies the `adapter_path` pointing to the saved LoRA checkpoint.
   - **Label Mapping:** Loads `used_label_text_map.json` from the checkpoint directory to know exactly which intent labels the model was fine-tuned on.
   - **Model Setup:** Uses Unsloth's `FastLanguageModel.from_pretrained` to load the base model and the LoRA adapters in 4-bit quantization, significantly speeding up inference. It also prepares the tokenizer with the correct chat template.

2. **`__call__(self, message)` (Prediction):**
   - **Prompt Formatting:** Wraps the user's input `message` inside a standard chat format (`{"role": "user", "content": ...}`) and applies the tokenizer's chat template.
   - **Generation:** Sends the tokenized prompt to the GPU (if available) and uses `.generate()` to predict the next tokens (the intent label).
   - **Post-processing:** Decodes the generated tokens into a string. It includes a fallback mechanism: if the model outputs something outside the predefined valid labels, it returns `"unknown_intent"` to ensure robust downstream handling.

### Hyperparameters & Training Details
In this project, experiments were conducted to find the optimal configuration. Below is the detailed hyperparameter setup utilized for the **Llama-3.2-3B-Instruct** baseline run (configured in `checkpoints/20260424_122036/train.yaml`):

#### 1. LOAD MODEL (Base Model Setup)
| Parameter | Value | Description |
|---------|----------|-------|
| `model_name` | `"unsloth/Llama-3.2-3B-Instruct"` | The base Large Language Model (LLM) used for fine-tuning. |
| `max_seq_length` | `128` | Maximum context window size. Limiting to 128 saves memory for short-sentence classification tasks. |
| `load_in_4bit` | `True` | Enables 4-bit quantization, drastically reducing the VRAM required to load the 3B model. |

#### 2. LORA HYPERPARAMETERS
| Parameter | Value | Description |
|---------|----------|-------|
| `lora_r` | `16` | Rank of the LoRA matrices. Determines the learning capacity of the adapter. |
| `lora_alpha` | `32` | Scaling factor for LoRA, typically set to twice `lora_r`. |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Applies LoRA to all Attention and MLP layers for a more comprehensive fine-tuning. |
| `lora_dropout` | `0` | Dropout rate. Set to 0 because Unsloth highly optimizes zero-dropout LoRA for maximum training speed. |
| `use_rslora` | `True` | Enables Rank-Stabilized LoRA, stabilizing the training process when scaling ranks. |
| `use_gradient_checkpointing` | `"unsloth"` | VRAM-saving technique that clears activations during the forward pass and recomputes them during the backward pass. |

#### 3. TRAINING ARGUMENTS
| Parameter | Value | Description |
|---------|----------|-------|
| `batch_size` | `2` | Number of data samples processed simultaneously per device. |
| `gradient_accumulation_steps` | `8` | Accumulates gradients over 8 steps, resulting in an effective batch size of 16. |
| `num_epochs` | `3` | Number of times the model iterates over the entire training dataset. |
| `learning_rate` | `2e-4` | Peak learning rate, a standard and effective value for LoRA fine-tuning. |
| `optimizer` | `"adamw_8bit"` | 8-bit version of the AdamW optimizer, saving ~75% of optimizer state memory. |
| `lr_scheduler_type` | `"cosine"` | Adjusts the learning rate following a cosine decay curve for smooth convergence. |
| `warmup_steps` | `50` | Gradually increases the learning rate from 0 to peak over the first 50 steps to avoid gradient shock. |
| `weight_decay` | `0.01` | L2 regularization penalty to prevent overfitting. |
| `early_stopping_patience` | `3` | Stops training early if `eval_loss` does not improve for 3 consecutive epochs. |

#### 4. SPECIAL TRAINING TECHNIQUES (Train On Response Only)
| Parameter | Value | Description |
|---------|----------|-------|
| `instruction_part` | `"<\|start_header_id\|>user<\|end_header_id\|>\n\n"` | The formatting tag for the user's prompt (Llama 3 standard). |
| `response_part` | `"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n"` | The formatting tag for the model's response (intent label). |
| **Masking Technique** | `train_on_responses_only` | Calculates the loss *only* on the assistant's answer, ignoring the long user prompt. This forces the model to focus entirely on label classification. |

## Evaluation Techniques & Model Selection

### 1. Zero-Shot System Prompt vs Internalized Knowledge
A key evaluation technique used in this project is the direct, rigorous comparison between the Fine-tuned model and its Base counterpart:
- **Base Model (Zero-Shot Prompting):** The base model is evaluated using a comprehensive **System Prompt** that explicitly lists all 36 available intent labels. It relies entirely on its zero-shot reasoning capabilities to pick the right label from the provided list in the prompt.
- **Fine-Tuned Model:** The fine-tuned model does *not* use the system prompt or the label list during inference. Because of the training process, it has **internalized the intent space**. It receives only the raw user message and outputs the correct label directly. 

This evaluation strategy clearly demonstrates that parameter-efficient fine-tuning (LoRA) significantly outperforms extensive prompt engineering for domain-specific classification tasks.

### 2. Model Selection Rationale
During our experimentation phase, including an alternative fine-tuning configuration using **Qwen2.5-3B**, the **Llama-3.2-3B-Instruct** model demonstrated better accuracy and training speed. Therefore, it was selected as the primary model to be evaluated on the unseen `Test` split to report the final benchmark results.

## Results

### Fine-tuned LLaMA-3.2-3B with LoRA (r=16, alpha=32)

| Metric | Val | Test |
|--------|-----|------|
| **Fine-tuned Accuracy** | 0.9523 | 0.9514 |
| **Base Model Accuracy** | 0.4415 | 0.4200 |
| **Delta (FT - Base)** | +0.5107 | +0.5313 |
| **Correct Samples** | 1018/1069 | 1017/1069 |

**Key Observations:**
- Val and test accuracies are nearly identical (< 1% difference), indicating **no overfitting**.
- The model generalizes well to unseen data.
- Fine-tuned model shows ~51% improvement over the zero-shot base model.
- Checkpoint: `checkpoints/20260424_122036/` (~95 MB adapter)



## Demo Video
[Watch here](YOUR_GOOGLE_DRIVE_LINK)

