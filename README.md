# Banking Intent Classification with Unsloth

## Setup
```bash
pip install -r requirements.txt
```

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