from threading import Thread
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)

from tunix.model import load_model

model_path = "/home/tanger/workspace/models/Qwen3-VL-4B-Instruct"

model, tokenizer, processor = load_model(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "test.png",
            },
            {
                "type": "text",
                "text": "描述这张图片。",
            },
        ],
    }
]

# build inputs
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

# streamer
streamer = TextIteratorStreamer(
    processor.tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

# generation kwargs
generation_kwargs = dict(
    **inputs,
    max_new_tokens=128,
    streamer=streamer,
)

# run generation in another thread
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# stream output
print("Assistant: ", end="", flush=True)

for new_text in streamer:
    print(new_text, end="", flush=True)

print()