import torch
from tunix.model import load_model

model_path = "/home/tanger/workspace/models/Qwen3-VL-4B-Instruct"

model, tokenizer = load_model(model_path)

model.eval()

# test question
messages = [
    {
        "role": "user",
        "content": "你是谁？请简单介绍一下自己。"
    }
]

# build prompt
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    # enable_thinking=False
)

print(f"text: {text}")

# tokenize
inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

# generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

# decode only generated part
response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=False
)

print("\nAssistant:")
print(response)