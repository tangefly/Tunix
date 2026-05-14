# train.py

import os
import json
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForImageTextToText,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

from data import SFTProcessor
from data import SFTDataset
from data import SFTCollator

# Config

MODEL_NAME = "/home/tanger/workspace/models/Qwen3.5-2B"
DATA_PATH = "/home/tanger/workspace/Tunix/data/robot/mllm_robot.json"
SAVE_PATH = "../result/qwen3_5_lora"
LOSS_PLOT_PATH = "./loss_curve.png"
DEVICE = "cuda"
MAX_LENGTH = 1024
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 5

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

processor = SFTProcessor(
    model_name=MODEL_NAME,
    max_length=MAX_LENGTH,
    image_base_dir=os.path.dirname(DATA_PATH),
)

tokenizer = processor.tokenizer

dataset = SFTDataset(
    data=data,
    processor=processor,
)

collator = SFTCollator(tokenizer)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)

model.to(DEVICE)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Disable cache for training
model.config.use_cache = False

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

loss_list = []
global_step = 0

for epoch in range(EPOCHS):

    progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    epoch_loss = 0.0
    for step, batch in enumerate(progress_bar):

        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        # Prepare multimodal inputs if present
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in batch:
            model_inputs["pixel_values"] = batch["pixel_values"].to(DEVICE)
        if "image_grid_thw" in batch:
            model_inputs["image_grid_thw"] = batch["image_grid_thw"].to(DEVICE)
        if "mm_token_type_ids" in batch:
            model_inputs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**model_inputs)
            loss = outputs.loss

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        current_loss = loss.item()
        epoch_loss += current_loss
        avg_loss = epoch_loss / (step + 1)
        loss_list.append(avg_loss)
        global_step += 1

        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "avg_loss": f"{avg_loss:.4f}",
        })

    print(f"\nEpoch {epoch + 1} Average Loss: {avg_loss:.4f}")

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"\nLoRA saved to: {SAVE_PATH}")

plt.figure(figsize=(10, 5))
plt.plot(loss_list)
plt.xlabel("Step")
plt.ylabel("Average Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig(LOSS_PLOT_PATH, dpi=300, bbox_inches="tight")
print(f"Loss curve saved to: {LOSS_PLOT_PATH}")
