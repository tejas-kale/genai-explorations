import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, get_scheduler
from transformers.generation import GenerationMixin


# HF Jobs injects this secret with `--secrets HF_TOKEN`.
# The same token must be able to read the private dataset and write the output model repo.
token = os.environ["HF_TOKEN"]

# Keep every setting overridable from `hf jobs run --env KEY=value`.
dataset_id = os.getenv("DATASET_ID", "tejaskale/florence-handwriting-ocr")
model_id = os.getenv("MODEL_ID", "microsoft/Florence-2-base-ft")
out_repo = os.getenv("OUT_REPO", "tejaskale/florence-handwriting-ocr-ft")
revision = os.getenv("FLORENCE_REVISION", "refs/pr/6")
epochs = int(os.getenv("EPOCHS", "10"))
lr = float(os.getenv("LR", "1e-6"))

# HF Jobs gives us CUDA on GPU flavours; this keeps the script runnable on CPU too.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Florence uses task tokens as prompts. `<OCR>` asks it to transcribe the image.
task = "<OCR>"

# The dataset was uploaded as a Hugging Face Dataset with columns:
#   image: decoded PIL image
#   text:  ground-truth transcript
items = list(load_dataset(dataset_id, token=token)["train"])

# Naive split: first 80% train, last 20% validation.
# Good enough for 30 images; no stratification or folds.
split = max(1, int(len(items) * 0.8))
train_items, val_items = items[:split], items[split:]

# Florence needs `trust_remote_code` because its model code lives in the model repo.
# `refs/pr/6` is pinned because newer/default remote code has broken in this setup.
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, revision=revision)

# `attn_implementation="eager"` tells Transformers to use the plain, older attention code.
# Attention is the part of the model that decides which image/text pieces to look at together.
# Faster attention paths exist, but Florence's custom code has hit compatibility errors with them.
# Eager is slower, but boring and reliable for this tiny 30-image run.
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision, attn_implementation="eager").to(device)

# Work around a current Transformers/Florence mismatch.
# Florence's outer model has a `generate()` method. Inside, it calls `generate()` again on its
# language model, the text-producing part. Newer Transformers expects that inner language model
# to explicitly inherit a helper class called GenerationMixin. Florence's downloaded code can
# generate text, but does not advertise that fact in the way newer Transformers expects.
# Without this, training works, but the smoke-test prediction crashes after training.
# This line creates a tiny temporary class that keeps the original Florence language model class
# and adds GenerationMixin to it. The object is still the same trained language model; we only
# add the missing generation helper methods that Transformers looks for.
model.language_model.__class__ = type("Florence2LanguageModel", (model.language_model.__class__, GenerationMixin), {})
# This line builds default generation settings from the language model config and attaches them.
# Without it, `generate()` can start but then fail because it cannot find those settings.
model.language_model.generation_config = GenerationConfig.from_model_config(model.language_model.config)

# The vision tower is the image-reading part of Florence. It turns pixels into internal image
# features, roughly like "this region has handwriting strokes, this region has whitespace".
# Freezing it means `requires_grad = False`: training will not change those image-reading weights.
# Trade-off: freezing uses less GPU memory, trains faster, and is less likely to overfit 30 images.
# Downside: if the handwriting/images are very different from what Florence already understands,
# the frozen image reader cannot adapt. Unfreezing may improve quality, but costs more memory and
# can overfit or crash on a T4. Start frozen; unfreeze only if the output is clearly image-blind.
for p in model.vision_tower.parameters():
    p.requires_grad = False


def collate(batch):
    # DataLoader gives us a small list of examples. Here batch size is 1, but this also works for more.
    # Convert every image to RGB because the processor/model expects normal three-channel colour images.
    images = [x["image"].convert("RGB") for x in batch]

    # `processor(...)` is Florence's input-preparation helper.
    # It does the image preprocessing: resize/normalise/turn PIL images into tensors.
    # It also tokenises the prompt text (`<OCR>`) into numbers the model can read.
    # `return_tensors="pt"` asks for PyTorch tensors; `padding=True` makes all items in a batch the same length.
    # `.to(device)` moves those tensors to the GPU when available.
    x = processor(text=[task] * len(batch), images=images, return_tensors="pt", padding=True).to(device)

    # Tokenise the expected transcripts separately as labels: these are the answers we want the model to learn.
    y = processor.tokenizer([x["text"] for x in batch], return_tensors="pt", padding=True).input_ids.to(device)

    # Padding tokens are filler tokens added to make sequences in the same batch equally long.
    # Example: ["hello", "a much longer sentence"] must become a rectangle of numbers, so "hello" gets fillers.
    # We do not want the model rewarded or punished for predicting filler tokens; they are not real text.
    # PyTorch cross-entropy loss uses `ignore_index=-100` by default, so any label set to -100 is skipped.
    # That is why pad token ids are replaced with -100 here.
    y[y == processor.tokenizer.pad_token_id] = -100

    # Florence expects the target transcript under the key `labels` when computing training loss.
    x["labels"] = y
    return x


# Batch size 1 is deliberate: tiny dataset, T4-friendly, no gradient accumulation ceremony.
train_loader = DataLoader(train_items, batch_size=1, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_items, batch_size=1, collate_fn=collate)

# AdamW is the optimiser: it nudges model weights in the direction that lowers loss.
# `lr` is the learning rate: the size of each nudge.
# If loss barely moves, try a larger value such as 3e-6 or 1e-5.
# If loss jumps around, gets worse, or output turns to junk, try smaller such as 3e-7.
# With only 30 images, change one thing at a time: lr first, then epochs, then unfreeze vision if needed.
optim = torch.optim.AdamW(model.parameters(), lr=lr)

# The scheduler changes the learning rate during training. `linear` starts at `lr` and steadily lowers it to zero.
# This makes early updates larger and later updates gentler, reducing the chance of overshooting near the end.
sched = get_scheduler("linear", optim, 0, epochs * len(train_loader))

# Main training loop. One pass over all training images is one epoch.
for epoch in range(epochs):
    # Put the model in training mode. Some layers behave differently while learning than while evaluating.
    model.train()

    # Keep every image's loss so we can print the average loss for this epoch.
    train_losses = []

    # `train_loader` yields one prepared batch at a time. Here each batch is one image and one transcript.
    for batch in train_loader:
        # Run the image and `<OCR>` prompt through the model, compare its predicted text tokens with `labels`,
        # and return one number: loss. Lower loss means the model's token predictions are closer to the transcript.
        loss = model(**batch).loss

        # Work backwards from the loss to calculate how each trainable weight contributed to the mistake.
        # These calculated "which weights should move which way" values are called gradients.
        loss.backward()

        # `optim.step()` changes the trainable weights using the gradients.
        # `sched.step()` updates the learning rate for the next step.
        # `optim.zero_grad()` clears old gradients so the next image starts fresh.
        optim.step(); sched.step(); optim.zero_grad()

        # Store a detached CPU copy of the loss for printing. Detach means "do not keep training history".
        train_losses.append(loss.detach().float().cpu())

    # Validation is only a loss sanity check, not a real OCR metric.
    # It tells us whether held-out examples are becoming easier for the model to predict.
    # It does not tell us exact character accuracy, word error rate, or whether the transcript is useful to a human.
    # For real OCR evaluation we would compute CER/WER on generated text, but that is intentionally skipped here.
    model.eval()
    val_losses = []

    # No learning during validation: save memory and avoid accidentally changing weights.
    with torch.no_grad():
        # Run each validation image through the same loss calculation, but do not call backward or optimiser steps.
        for batch in val_loader:
            val_losses.append(model(**batch).loss.detach().float().cpu())

    # Print average train/validation loss so the HF job log shows whether training is moving in the right direction.
    print({"epoch": epoch + 1, "train_loss": float(sum(train_losses) / len(train_losses)), "val_loss": float(sum(val_losses) / len(val_losses))})

# Smoke test: run one validation image through the trained model and print expected vs predicted.
sample = val_items[0] if val_items else train_items[0]
inputs = processor(text=task, images=sample["image"].convert("RGB"), return_tensors="pt").to(device)

# Greedy generation avoids Florence beam-search/cache bugs seen with this dependency combo.
ids = model.generate(**inputs, max_new_tokens=128, num_beams=1, do_sample=False, use_cache=False)
predicted = processor.batch_decode(ids, skip_special_tokens=True)[0]
print({"expected": sample["text"], "predicted": predicted})

# Save the fine-tuned model and processor together so inference can load one repo.
model.push_to_hub(out_repo, private=True, token=token)
processor.push_to_hub(out_repo, private=True, token=token)
