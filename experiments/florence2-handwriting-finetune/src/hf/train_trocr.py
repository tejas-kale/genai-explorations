import json
import os
import random
import re

import jiwer
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# HF Jobs injects this secret with `--secrets HF_TOKEN`.
# The same token must be able to read the private dataset and write the output model repo.
token = os.environ["HF_TOKEN"]

# Keep every setting overridable from `hf jobs run --env KEY=value`.
dataset_id = os.getenv("DATASET_ID", "tejaskale/florence-handwriting-ocr")
model_id = os.getenv("MODEL_ID", "microsoft/trocr-base-handwritten")
out_repo = os.getenv("OUT_REPO", "tejaskale/trocr-handwriting-ocr-ft")
epochs = int(os.getenv("EPOCHS", "100"))
lr = float(os.getenv("LR", "5e-5"))
batch_size = int(os.getenv("BATCH_SIZE", "2"))
max_length = int(os.getenv("MAX_LENGTH", "512"))
train_frac = float(os.getenv("TRAIN_FRAC", "0.7"))
split_seed = int(os.getenv("SPLIT_SEED", "42"))

# HF Jobs gives us CUDA on GPU flavours; this keeps the script runnable on CPU too.
device = "cuda" if torch.cuda.is_available() else "cpu"

# The dataset was uploaded as a Hugging Face Dataset with columns:
#   image: decoded PIL image
#   text:  ground-truth transcript
#   image_id: filename stem
items = list(load_dataset(dataset_id, token=token)["train"])

# Deterministic 70/30 split with a fixed seed. This experiment is deliberately about
# overfitting: we train on 70% of the pairs and later evaluate on that SAME 70% to see
# how hard a small VisionEncoderDecoder model can memorise full-page handwriting
# transcripts. The other 30% is held out and untouched by this script.
rng = random.Random(split_seed)
order = list(range(len(items)))
rng.shuffle(order)
split = max(1, int(len(order) * train_frac))
train_items = [items[i] for i in order[:split]]
holdout_items = [items[i] for i in order[split:]]
print({"num_total": len(items), "num_train": len(train_items), "num_holdout": len(holdout_items)})

# TrOCR is a plain VisionEncoderDecoderModel (a ViT-style image encoder feeding a
# RoBERTa-style text decoder) paired with TrOCRProcessor. Unlike Florence-2, there is no
# `trust_remote_code`, no pinned revision, and no generation-mixin workarounds needed:
# TrOCR's classes are fully supported by stock Transformers.
processor = TrOCRProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)

# TrOCR's decoder does not know on its own which token starts a sequence, which one is
# padding, or which one ends generation. The processor's tokenizer is RoBERTa-based, so
# `cls_token` doubles as "start" and `sep_token` doubles as "end". These must be copied
# onto both the model config (used during the loss/forward pass) and the generation
# config (used by `generate()`), otherwise training works but generation misbehaves.
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size
# NOTE: only `generation_config` may carry generation-strategy knobs like `max_length` in
# current Transformers; setting `model.config.max_length` directly raises a ValueError
# ("this strategy to control generation is not supported anymore"). Architecture-level
# ids (decoder_start/pad/eos token, vocab_size) still belong on `model.config` because
# the encoder-decoder forward pass (label shifting) reads them from there.
model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
model.generation_config.max_length = max_length


def collate(batch):
    # Convert every image to RGB: the processor's image side expects normal
    # three-channel colour images and always resizes/normalises to a fixed square size,
    # so unlike Florence there is no need for `padding=True` on the pixel side.
    images = [x["image"].convert("RGB") for x in batch]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)

    # Tokenise the full-page transcripts as labels, fixed to `max_length` tokens.
    # Padding to a fixed length (rather than the batch's longest) keeps shapes stable
    # across batches, which matters less here than staying close to the 512-token cap
    # that comfortably covers the longest transcript (~1500 chars).
    labels = processor.tokenizer(
        [x["text"] for x in batch],
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids

    # Padding tokens are filler, not real text. Cross-entropy loss ignores label
    # positions set to -100, so replace pad token ids with -100 before handing labels
    # to the model, exactly as in the Florence script.
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": pixel_values, "labels": labels.to(device)}


# Small batch, shuffled every epoch. Batch size is an overfitting knob here, not a
# throughput one: this dataset is ~20 images.
train_loader = DataLoader(train_items, batch_size=batch_size, shuffle=True, collate_fn=collate)

# Overfitting is the deliberate goal of this experiment, so the optimiser setup is kept
# intentionally plain: a constant learning rate (no scheduler, no warmup/decay) and no
# weight decay. Anything that would normally fight overfitting is switched off.
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

for epoch in range(epochs):
    model.train()
    train_losses = []
    for batch in train_loader:
        loss = model(**batch).loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        train_losses.append(loss.detach().float().cpu())
    print({"epoch": epoch + 1, "train_loss": float(sum(train_losses) / len(train_losses))})


def normalize(value):
    # lowercase, collapse all whitespace (incl. newlines) to single spaces, strip.
    return re.sub(r"\s+", " ", value.lower()).strip()


# Evaluate CER/WER by generating on the SAME 70% training images used above. This is not
# a generalisation check: it directly measures how well the model has memorised the
# training transcripts, which is the point of this experiment.
model.eval()
results = {}
hyp_norm_all, ref_norm_all = [], []
with torch.no_grad():
    for item in train_items:
        pixel_values = processor(images=item["image"].convert("RGB"), return_tensors="pt").pixel_values.to(device)
        ids = model.generate(pixel_values, max_length=max_length, num_beams=1, do_sample=False)
        predicted = processor.batch_decode(ids, skip_special_tokens=True)[0]
        reference = item["text"]

        hyp_norm = normalize(predicted)
        ref_norm = normalize(reference)
        image_cer = jiwer.cer(ref_norm, hyp_norm)
        image_wer = jiwer.wer(ref_norm, hyp_norm)
        print({"image_id": item["image_id"], "cer": image_cer, "wer": image_wer})

        results[item["image_id"]] = {
            "predicted": predicted,
            "reference": reference,
            "cer": image_cer,
            "wer": image_wer,
        }
        hyp_norm_all.append(hyp_norm)
        ref_norm_all.append(ref_norm)

overall_cer = jiwer.cer(ref_norm_all, hyp_norm_all)
overall_wer = jiwer.wer(ref_norm_all, hyp_norm_all)
print({"overall_cer": overall_cer, "overall_wer": overall_wer})

output = {
    "model_id": model_id,
    "num_train": len(train_items),
    "num_holdout": len(holdout_items),
    "train_image_ids": [x["image_id"] for x in train_items],
    "images": results,
    "overall_cer": overall_cer,
    "overall_wer": overall_wer,
}
results_path = "trocr_train_eval.json"
with open(results_path, "w") as f:
    json.dump(output, f, indent=2)

# Save the fine-tuned model and processor together so inference can load one repo, and
# push the CER/WER results alongside them so the memorisation numbers travel with the model.
model.push_to_hub(out_repo, private=True, token=token)
processor.push_to_hub(out_repo, private=True, token=token)
HfApi().upload_file(
    path_or_fileobj=results_path,
    path_in_repo=results_path,
    repo_id=out_repo,
    token=token,
)
