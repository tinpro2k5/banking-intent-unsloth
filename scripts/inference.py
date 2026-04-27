# scripts/inference.py
import argparse
import json
import warnings
from pathlib import Path

import yaml
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\.modeling_attn_mask_utils",
)


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

        adapter_path = resolve_path(cfg["adapter_path"])
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        if adapter_path.is_dir() and not has_adapter_files(adapter_path) and any(item.is_dir() for item in adapter_path.iterdir()):
            raise ValueError(
                f"adapter_path must point to a saved checkpoint directory, not a root folder with multiple runs: {adapter_path}"
            )

        used_label_map_name = cfg.get("used_label_text_map", "used_label_text_map.json")
        # Strip any leading directory prefix (e.g. "configs/") so the lookup
        # matches the bare filename that train.py actually saves inside the checkpoint.
        used_label_file = Path(used_label_map_name).name
        checkpoint_used_label_map = adapter_path / used_label_file
        if checkpoint_used_label_map.exists():
            used_label_map_path = checkpoint_used_label_map
        else:
            # Fall back to the repo-level path from config, then full label map.
            used_label_map_path = resolve_path(used_label_map_name)
            if not used_label_map_path.exists():
                used_label_map_path = resolve_path(cfg["label_text_map"])

        with open(used_label_map_path, "r", encoding="utf-8") as f:
            id2label_raw = json.load(f)
        self.id2label = {int(k): v for k, v in id2label_raw.items()}
        self.valid_labels = set(self.id2label.values())

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
        if predicted_label not in self.valid_labels:
            return "unknown_intent"
        return predicted_label


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive banking intent inference")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "inference.yaml"),
        help="Path to inference YAML config",
    )
    return parser.parse_args()


def run_interactive_loop(clf: IntentClassification):
    print("=" * 72)
    print("Banking Intent Inference (Interactive)")
    print("=" * 72)
    print("How to run:")
    print("  python scripts/inference.py")
    print("  python scripts/inference.py --config configs/inference.yaml")
    print("Type your message and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print(f"Loaded intents: {len(clf.valid_labels)}")
    for idx, label in enumerate(sorted(clf.valid_labels), start=1):
        print(f"{idx:>2}. {label}")
    print("-" * 72)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive inference.")
            break

        if not user_input:
            print("\n[Input] Please enter a non-empty message.")
            print("-" * 72)
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting interactive inference.")
            break

        predicted = clf(user_input)
        print("\n[Result]")
        print(f"Message         : {user_input}")
        print(f"Predicted intent: {predicted}")
        print("-" * 72)


if __name__ == "__main__":
    args = parse_args()
    clf = IntentClassification(args.config)
    run_interactive_loop(clf)