# scripts/inference.py
import yaml, json, re
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from transformers import TextStreamer


class IntentClassification:
    def __init__(self, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        with open(cfg["label_text_map"]) as f:
            label2id = json.load(f)
        self.id2label = {v: k for k, v in label2id.items()}

        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["model_path"],
            max_seq_length=cfg["max_seq_length"],
            dtype=cfg["dtype"],
            load_in_4bit=cfg["load_in_4bit"],
        )
        self.tokenizer = get_chat_template(self.tokenizer, chat_template=cfg["chat_template"])
        FastLanguageModel.for_inference(self.model) # Enable native x2 faster inference
        self.max_seq_length = cfg["max_seq_length"]
        self.max_new_tokens = cfg["max_new_tokens"]
        self.text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)

    def __call__(self, message: str) -> str:
        messages = [
            {"role": "user", "content": f"Classify the intent: {message}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=20, use_cache=True)
        decoded = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Model outputs intent string directly
        predicted = decoded.strip().split("\n")[0]
        
        input_ids = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors = "pt"
        )["input_ids"].to("cuda")
        _ = self.model.generate(
            
            streamer = self.text_streamer,
            max_new_tokens = self.max_new_tokens,
            pad_token_id = self.tokenizer.eos_token_id
        )
        return predicted


# Usage example
if __name__ == "__main__":
    clf = IntentClassification("configs/inference.yaml")
    msg = "I lost my card and need a replacement."
    print(f"Message : {msg}")
    print(f"Predicted Intent: {clf(msg)}")