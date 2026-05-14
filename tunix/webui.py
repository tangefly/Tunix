import gc
import re
from threading import Thread

import gradio as gr
import torch

from peft import PeftModel
from transformers import (
    TextIteratorStreamer,
)

from tunix.model import load_model

model = None
tokenizer = None
processor = None
current_lora = None
is_multimodal = False


THINK_TEMPLATE = """
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

def check_multimodal(model, processor):

    if hasattr(model.config, "vision_config"):
        return True

    if hasattr(processor, "image_processor"):
        return True

    if hasattr(processor, "video_processor"):
        return True

    return False

def _load_model(model_path):

    global model
    global tokenizer
    global processor
    global current_lora
    global is_multimodal

    try:

        _offload_model()

        print(f"Loading model from: {model_path}")

        model, tokenizer, processor = load_model(model_path)

        model = model.eval()

        current_lora = None

        is_multimodal = check_multimodal(
            model,
            processor
        )

        return (
            f"✅ Base model loaded: {model_path}",
            gr.update(visible=is_multimodal),
            gr.update(visible=is_multimodal)
        )

    except Exception as e:

        return (
            f"❌ Load failed: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=False)
        )

def _load_lora(lora_path):

    global model
    global current_lora

    try:

        if model is None:
            return "⚠️ 请先加载 base model"

        if lora_path == "":
            return "⚠️ lora 文件路径不能为空"

        print(f"Loading LoRA from: {lora_path}")

        if current_lora is not None:

            model = (
                model.unload()
                if hasattr(model, "unload")
                else model
            )

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

    global model
    global processor
    global current_lora
    global is_multimodal

    try:

        if model is not None:
            del model

        if processor is not None:
            del processor

        model = None
        processor = None
        current_lora = None
        is_multimodal = False

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return "🧹 Model offloaded"

    except Exception as e:

        return f"❌ Offload failed: {str(e)}"


def build_messages(history, message, image, video):

    prompt_messages = []

    for item in history:
        if item["role"] == "assistant":
            prompt_messages.append({
                "role": "assistant",
                "content": remove_details(item["content"])
            })
        elif item["role"] == "user":
            prompt_messages.append(item)

    if image is not None or video is not None:
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if video is not None:
            content.append({"type": "video", "video": video})
        content.append({"type": "text", "text": message})

        print(content)

        prompt_messages.append({
            "role": "user",
            "content": content
        })
    else:
        prompt_messages.append({
            "role": "user",
            "content": message
        })

    return prompt_messages


def chat_fn(message, image, video, history):

    global model
    global processor

    if model is None or tokenizer is None:
        yield history + [
            {
                "role": "assistant",
                "content": "⚠️ 请先加载模型！"
            }
        ]
        return

    prompt_messages = build_messages(history, message, image, video)

    if is_multimodal:
        text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        processor_kwargs = {"text": [text], "return_tensors": "pt"}
        if image is not None:
            processor_kwargs["images"] = [image]
        if video is not None:
            processor_kwargs["videos"] = [video]

        inputs = processor(**processor_kwargs).to(model.device)
    else:
        text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

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
        thread = Thread(
            target=model.generate,
            kwargs=generation_kwargs
        )
        thread.start()

        full_generated_text = ""
        output_generated_text = ""

        current_history = history + [
            {
                "role": "user",
                "content": message
            }
        ]

        for new_text in streamer:
            full_generated_text += new_text
            if "</think>" in full_generated_text:
                think_context, answer = extract_think(full_generated_text)
                think_context = THINK_TEMPLATE.format(think=think_context)
                output_generated_text = (think_context + "\n" + answer)
            elif "<think>" in full_generated_text:
                output_generated_text = ("Thinking ...\n\n" + full_generated_text.replace("<think>", ""))
            else:
                output_generated_text = full_generated_text

            yield current_history + [
                {
                    "role": "assistant",
                    "content": output_generated_text
                }
            ]

        thread.join()

with gr.Blocks(title="LLM WebUI", fill_height=True) as demo:

    gr.Markdown("# 🧠 LLM Chat WebUI")

    with gr.Row():

        with gr.Column(scale=3):
            with gr.Row():
                model_path_input = gr.Textbox(label="Base Model Path", placeholder="/path/to/base-model")
                lora_path_input = gr.Textbox(label="LoRA Path (optional)", placeholder="/path/to/lora-adapter")
            status_box = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Column(scale=1):
            load_btn = gr.Button("Load Base", variant="primary")
            load_lora_btn = gr.Button("Load LoRA")
            offload_btn = gr.Button("Offload", variant="secondary")

    gr.Markdown("---")

    with gr.Row():

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=500, autoscroll=False)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="请输入内容...",
                    lines=1,
                    max_lines=6,
                    show_label=False,
                    autoscroll=False
                )

        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Image",
                visible=False,
                height=256,
                show_download_button=False,
                show_fullscreen_button=False
            )

            video_input = gr.Video(
                label="Video",
                visible=False,
                interactive=True,
                height=256,
                show_download_button=False,
            )

    prompt_state = gr.State("")

    def clear_chat():
        return ([], "", None)
    
    def prepare_prompt(prompt):
        return prompt.strip(), ""

    msg_input.submit(
        fn=prepare_prompt,
        inputs=msg_input,
        outputs=[prompt_state, msg_input]
    ).then(
        fn=chat_fn,
        inputs=[prompt_state, image_input, video_input, chatbot],
        outputs=chatbot
    )
    load_btn.click(fn=_load_model, inputs=model_path_input, outputs=[status_box, image_input, video_input])
    load_lora_btn.click(fn=_load_lora, inputs=lora_path_input, outputs=status_box)
    offload_btn.click(fn=_offload_model, outputs=status_box)

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )