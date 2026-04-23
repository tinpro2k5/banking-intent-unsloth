from sklearn.metrics import accuracy_score

def predict(text):
    prompt = f"Classify the banking intent of this message.\nMessage: {text}\nIntent:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=20)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded.split("Intent:")[-1].strip().split("\n")[0]

FastLanguageModel.for_inference(model)
preds = [predict(row["text"]) for _, row in test_df.iterrows()]
true_labels = [id2label[row["label"]] for _, row in test_df.iterrows()]
print(f"Test Accuracy: {accuracy_score(true_labels, preds):.4f}")


