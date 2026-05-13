from threading import Thread
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)

from tunix.model import load_model

model_path = "/home/xiaoxunpeng/workspace/models/Qwen/Qwen3-VL-4B-Instruct"

model, tokenizer, processor = load_model(model_path)

messages1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/tanger/workspace/Tunix/data/robot/images/ameca_9.png",
            },
            {
                "type": "image",
                "image": "/home/tanger/workspace/Tunix/data/robot/images/leju_kuafu_2.png",
            },
            {
                "type": "text",
                "text": "分析这两张图片！",
            },
        ],
    }
]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "hello",
            },
        ],
    }
]

# build inputs
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt",
# )

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids

inputs = tokenizer(text, add_special_tokens=False).input_ids

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
    max_new_tokens=40960,
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