import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, get_scheduler
from transformers.generation import GenerationMixin


token = os.environ["HF_TOKEN"]
dataset_id = os.getenv("DATASET_ID", "tejaskale/florence-handwriting-ocr")
model_id = os.getenv("MODEL_ID", "microsoft/Florence-2-base-ft")
out_repo = os.getenv("OUT_REPO", "tejaskale/florence-handwriting-ocr-ft")
revision = os.getenv("FLORENCE_REVISION", "refs/pr/6")
epochs = int(os.getenv("EPOCHS", "3"))
lr = float(os.getenv("LR", "1e-6"))
device = "cuda" if torch.cuda.is_available() else "cpu"
task = "<OCR>"

items = list(load_dataset(dataset_id, token=token)["train"])
split = max(1, int(len(items) * 0.8))
train_items, val_items = items[:split], items[split:]

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision, attn_implementation="eager").to(device)
model.language_model.__class__ = type("Florence2LanguageModel", (model.language_model.__class__, GenerationMixin), {})
model.language_model.generation_config = GenerationConfig.from_model_config(model.language_model.config)
for p in model.vision_tower.parameters():
    p.requires_grad = False


def collate(batch):
    images = [x["image"].convert("RGB") for x in batch]
    x = processor(text=[task] * len(batch), images=images, return_tensors="pt", padding=True).to(device)
    y = processor.tokenizer([x["text"] for x in batch], return_tensors="pt", padding=True).input_ids.to(device)
    y[y == processor.tokenizer.pad_token_id] = -100
    x["labels"] = y
    return x


train_loader = DataLoader(train_items, batch_size=1, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_items, batch_size=1, collate_fn=collate)
optim = torch.optim.AdamW(model.parameters(), lr=lr)
sched = get_scheduler("linear", optim, 0, epochs * len(train_loader))

for epoch in range(epochs):
    model.train()
    train_losses = []
    for batch in train_loader:
        loss = model(**batch).loss
        loss.backward()
        optim.step(); sched.step(); optim.zero_grad()
        train_losses.append(loss.detach().float().cpu())
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            val_losses.append(model(**batch).loss.detach().float().cpu())
    print({"epoch": epoch + 1, "train_loss": float(sum(train_losses) / len(train_losses)), "val_loss": float(sum(val_losses) / len(val_losses))})

sample = val_items[0] if val_items else train_items[0]
inputs = processor(text=task, images=sample["image"].convert("RGB"), return_tensors="pt").to(device)
ids = model.generate(**inputs, max_new_tokens=128, num_beams=1, do_sample=False, use_cache=False)
predicted = processor.batch_decode(ids, skip_special_tokens=True)[0]
print({"expected": sample["text"], "predicted": predicted})

model.push_to_hub(out_repo, private=True, token=token)
processor.push_to_hub(out_repo, private=True, token=token)
