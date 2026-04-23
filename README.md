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

## Inference
```bash
bash inference.sh
```

## Dataset
BANKING77 — 77 banking intent classes. We sample 20 classes for this project.

## Model
Fine-tuned LLaMA-3-8B with 4-bit quantization via Unsloth + LoRA.

## Results
| Split | Accuracy |
|-------|----------|
| Test  | XX%      |



Model Checkpoint

Due to size limitations, the trained model checkpoint is not included in this repository.

You can either:

Train the model using train.py
Or use the sample checkpoint provided via Google Drive (link below)


## Demo Video
[Watch here](YOUR_GOOGLE_DRIVE_LINK)