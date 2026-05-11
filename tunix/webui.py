import gradio as gr
import torch
import gc
import re
from threading import Thread
from transformers import TextIteratorStreamer
from peft import PeftModel

from model import load_model

model = None
tokenizer = None
current_lora = None

Think_Template = """
<details>
<summary>🧠 思考过程</summary>
{think}
</details>
"""

def remove_details(text):
    new_text = re.sub(r"<details>.*?</details>", "", text, flags=re.S)
    return new_text.strip()

def extract_think(text):

    think = ""

    if "<think>" in text and "</think>" in text:
        match = re.search(r"<think>(.*?)</think>", text, re.S)
        if match:
            think = match.group(1).strip()

        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    elif "</think>" in text:
        parts = text.split("</think>", 1)
        think = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else ""

    else:
        answer = text.strip()

    return think, answer

def _load_model(model_path):
    global model, tokenizer, current_lora

    try:
        _offload_model()

        print(f"Loading model from: {model_path}")

        model, tokenizer = load_model(model_path)

        model = model.eval()
        current_lora = None

        return f"✅ Base model loaded: {model_path}"

    except Exception as e:
        return f"❌ Load failed: {str(e)}"

def _load_lora(lora_path):
    global model, current_lora

    try:
        if model is None:
            return "⚠️ 请先加载 base model"
        
        if lora_path == "":
            return "⚠️ lora 文件路径不能为空" 

        print(f"Loading LoRA from: {lora_path}")

        if current_lora is not None:
            model = model.unload() if hasattr(model, "unload") else model

        model = PeftModel.from_pretrained(
            model,
            lora_path,
            is_trainable=False
        )

        current_lora = lora_path

        return f"✅ LoRA loaded: {lora_path}"

    except Exception as e:
        return f"❌ LoRA load failed: {str(e)}"

def _offload_model():
    global model, tokenizer, current_lora

    try:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer

        model = None
        tokenizer = None
        current_lora = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return "🧹 Model offloaded"

    except Exception as e:
        return f"❌ Offload failed: {str(e)}"

def chat_fn(message, history):
    global model, tokenizer

    if model is None or tokenizer is None:
        yield "⚠️ 请先加载模型！"
        return

    prompt_messages = []
    for item in history:
        if item["role"] == "assistant":
            prompt_messages.append({
                "role": "assistant",
                "content": remove_details(item["content"])
            })
        elif item["role"] == "user":
            prompt_messages.append(item)

    prompt_messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    with torch.inference_mode():
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        full_generated_text = ""
        output_generated_text = ""

        for new_text in streamer:
            full_generated_text += new_text

            if "</think>" in full_generated_text:
                think_context, full_generated_text = extract_think(full_generated_text)
                think_context = Think_Template.format(think=think_context)
                full_generated_text = think_context + "\n" + full_generated_text
                output_generated_text = full_generated_text

            elif "<think>" in full_generated_text:
                output_generated_text = "Thinking ...\n\n" + full_generated_text.replace("<think>", "")

            else:
                output_generated_text = full_generated_text

            yield output_generated_text

        thread.join()

with gr.Blocks(title="LLM WebUI", fill_height=True) as demo:
    gr.Markdown("# 🧠 LLM Chat WebUI")

    with gr.Row():

        with gr.Column(scale=3):
            model_path_input = gr.Textbox(
                label="Base Model Path",
                placeholder="/path/to/base-model"
            )

            lora_path_input = gr.Textbox(
                label="LoRA Path (optional)",
                placeholder="/path/to/lora-adapter"
            )

            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=1
            )

        with gr.Column(scale=1):
            load_btn = gr.Button("Load Base", variant="primary")
            load_lora_btn = gr.Button("Load LoRA")
            offload_btn = gr.Button("Offload", variant="secondary")

    gr.Markdown("---")

    # Chat
    gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        fill_height=True,
        autoscroll=False
    )

    load_btn.click(_load_model, inputs=model_path_input, outputs=status_box)
    load_lora_btn.click(_load_lora, inputs=lora_path_input, outputs=status_box)
    offload_btn.click(_offload_model, outputs=status_box)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)