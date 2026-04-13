# scripts/inference.py
import yaml, json, re
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        with open(cfg["label_map"]) as f:
            label2id = json.load(f)
        self.id2label = {v: k for k, v in label2id.items()}

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["model_path"],
            max_seq_length=cfg["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        self.max_seq_length = cfg["max_seq_length"]

    def __call__(self, message: str) -> str:
        prompt = f"Classify the banking intent of this message.\nMessage: {message}\nIntent:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=20, use_cache=True)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract intent from after "Intent:"
        predicted = decoded.split("Intent:")[-1].strip().split("\n")[0]
        return predicted


# Usage example
if __name__ == "__main__":
    clf = IntentClassification("configs/inference.yaml")
    msg = "I lost my card and need a replacement."
    print(f"Message : {msg}")
    print(f"Predicted Intent: {clf(msg)}")