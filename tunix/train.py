# train.py

import json
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

from .data.preprocess import SFTProcessor
from .data.dataset import SFTDataset
from .data.collator import SFTCollator

# Config

MODEL_NAME = "/home/tanger/workspace/models/Qwen3-4B"
DATA_PATH = "/home/tanger/workspace/Tunix/data/ruozhiba.json"
SAVE_PATH = "./qwen3_lora"
LOSS_PLOT_PATH = "./loss_curve.png"
DEVICE = "cuda"
MAX_LENGTH = 2048
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 5

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

processor = SFTProcessor(
    model_name=MODEL_NAME,
    max_length=MAX_LENGTH,
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

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

model.to(DEVICE)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# 关闭 cache
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

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
plt.savefig(LOSS_PLOT_PATH, dpi=300, bbox_inches="tight",)
print(f"Loss curve saved to: {LOSS_PLOT_PATH}")