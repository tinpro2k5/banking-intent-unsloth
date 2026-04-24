from sklearn.metrics import accuracy_score

def predict(text):
    messages = [{"role": "user", "content": f"Classify the intent: {text}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip().split("\n")[0]

FastLanguageModel.for_inference(model)
preds = [predict(row["text"]) for _, row in test_df.iterrows()]
true_labels = [id2label[row["label"]] for _, row in test_df.iterrows()]
print(f"Test Accuracy: {accuracy_score(true_labels, preds):.4f}")


