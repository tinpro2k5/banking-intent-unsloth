# scripts/inference.py
import json
from pathlib import Path

import yaml
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    candidates = [candidate, REPO_ROOT / candidate]
    for item in candidates:
        if item.exists():
            return item.resolve()
    return (REPO_ROOT / candidate).resolve()


def has_adapter_files(path: Path) -> bool:
    return any(
        (path / filename).exists()
        for filename in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
        )
    )


class IntentClassification:
    def __init__(self, model_path):
        with open(model_path) as f:
            cfg = yaml.safe_load(f)

        label_map_path = resolve_path(cfg["label_text_map"])
        with open(label_map_path) as f:
            label2id = json.load(f)
        self.id2label = {v: k for k, v in label2id.items()}

        adapter_path = resolve_path(cfg["adapter_path"])
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        if adapter_path.is_dir() and not has_adapter_files(adapter_path) and any(item.is_dir() for item in adapter_path.iterdir()):
            raise ValueError(
                f"adapter_path must point to a saved checkpoint directory, not a root folder with multiple runs: {adapter_path}"
            )

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=cfg["max_seq_length"],
            dtype=cfg.get("dtype", None),
            load_in_4bit=cfg.get("load_in_4bit", True),
        )
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_length = None
        self.tokenizer = get_chat_template(self.tokenizer, chat_template=cfg["chat_template"])
        FastLanguageModel.for_inference(self.model) # Enable native x2 faster inference
        self.max_seq_length = cfg["max_seq_length"]
        self.max_new_tokens = cfg["max_new_tokens"]
        # self.text_streamer = TextStreamer(self.tokenizer, skip_prompt=True) # Optional: for streaming generation output

    def __call__(self, message: str) -> str:
        messages = [{"role": "user", "content": f"Classify the intent: {message}"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        outputs = self.model.generate(
            **inputs, # python unpacking to pass input_ids, attention_mask, etc.
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            # streamer=self.streamer, use return values instead of streaming for better control
        )

        decoded = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        predicted_label = decoded.strip().split("\n")[0].strip()
        return predicted_label
        

# Usage example
if __name__ == "__main__":
    clf = IntentClassification(str(REPO_ROOT / "configs" / "inference.yaml"))
    msg = "I lost my card and need a replacement."
    print(f"Message : {msg}")
    print(f"Predicted Intent: {clf(msg)}")