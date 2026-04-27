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

## Prepare Data
```bash
python scripts/preprocess_data.py
```

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
- Loads the configuration from `configs/inference.yaml`
- Resolves a model checkpoint automatically from the latest timestamped run under `checkpoints/`
- Applies the same chat template used during training
- Uses `sample_data/val.csv` by default, but you can override the split when needed
- Generates one intent label for the fine-tuned checkpoint and one for the base model
- Compares both against the ground-truth labels and prints the accuracy gap

Useful modes:
- `--split val|test|train` selects which split to evaluate, default is `val`
- `--best-from-run` uses the best checkpoint recorded in `trainer_state.json`
- `--model-path ...` evaluates a specific checkpoint or Hugging Face model id

Examples:
```bash
# Evaluate the latest run on the validation split and compare it with the base model
bash evaluate.sh

# Evaluate the latest run on the test split
bash evaluate.sh --split test

# Evaluate a specific run directory and use its best checkpoint
bash evaluate.sh --run-dir checkpoints/20260424_231500 --best-from-run

# Evaluate the base model directly
bash evaluate.sh --model-path unsloth/Llama-3.2-3B-Instruct
```

## Inference
```bash
bash inference.sh
```

## Dataset
BANKING77 — 77 banking intent classes. This project fine-tunes on a subset of the full label space.

## Model
Fine-tuned LLaMA-3-3B Instruct with 4-bit quantization via Unsloth + LoRA.

### Hyperparameters & Training Details
As per the project requirements, the following hyperparameters and techniques were used during fine-tuning (configured in `configs/train.yaml`):

- **Batch Size:** `2` (with gradient accumulation steps = 8)
- **Learning Rate:** `2e-4` (Cosine scheduler, 50 warmup steps)
- **Optimizer:** `adamw_8bit`
- **Number of Epochs:** `3`
- **Maximum Sequence Length:** `128`
- **Regularization & Techniques:** 
  - Weight Decay: `0.01`
  - LoRA specific: `r=16`, `alpha=32`, `dropout=0`
  - Early Stopping: Patience of `3` epochs
  - Instruction tuning: Trained only on assistant responses (`train_on_responses_only` unsloth utility)

## Results
| Split | Accuracy |
|-------|----------|
| Val   | See evaluate output |



Model Checkpoint



## Demo Video
[Watch here](YOUR_GOOGLE_DRIVE_LINK)
- Preprocessing generates `configs/used_label_text_map.json`, and each checkpoint copies that file so inference and evaluation use the trained label subset instead of the full BANKING77 label space.
