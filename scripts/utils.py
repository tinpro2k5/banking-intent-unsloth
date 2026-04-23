
def apply_chat_template(tokenizer, messages, tokenize=True, add_generation_prompt=True):
    """Applies a chat template to the input messages.
    Args:
        tokenizer: The tokenizer to use for formatting.
        messages: A list of dictionaries with "role" and "content" keys.
        tokenize: Whether to return tokenized output or raw text.
        add_generation_prompt: Whether to add a generation prompt at the end for the assistant's response.
    Returns:
        Formatted text or tokenized output depending on the `tokenize` flag.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt
    )

def formatting_prompts_func(examples, tokenizer=None) -> dict:
    """Formats the input examples into a prompt format suitable for training.
    Each example is transformed into a chat format where the user input is the text to classify,"""
    inputs  = examples["text"]
    outputs = examples["label"]

    texts = []
    for inp, out in zip(inputs, outputs):
       
        messages = [
            {"role": "user", "content": f"Classify the intent: {inp}"},
            {"role": "assistant", "content": out}
        ]

        texts.append(apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=False))

    return {"text": texts}
