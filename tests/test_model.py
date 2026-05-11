import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/home/tanger/workspace/models/Falcon-H1-7B-Instruct"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

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
)

# tokenize
inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

# generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
    )

# decode only generated part
response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)

print("\nAssistant:")
print(response)