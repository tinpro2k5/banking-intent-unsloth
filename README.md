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
bash run_evaluate.sh

# Evaluate the latest run on the test split
bash run_evaluate.sh --split test

# Evaluate a specific run directory and use its best checkpoint
bash run_evaluate.sh --run-dir checkpoints/20260424_231500 --best-from-run

# Evaluate the base model directly
bash run_evaluate.sh --model-path unsloth/Llama-3.2-3B-Instruct
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

Due to size limitations, the trained model checkpoint is not included in this repository.

You can either:

Train the model using `scripts/train.py`
Or use the sample checkpoint generated under `checkpoints/<timestamp>/`


## Demo Video
[Watch here](YOUR_GOOGLE_DRIVE_LINK)