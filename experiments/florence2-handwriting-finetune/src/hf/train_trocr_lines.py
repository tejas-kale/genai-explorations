import json
import os
import re

import jiwer
import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler


# HF Jobs injects this secret with `--secrets HF_TOKEN`.
# The same token must be able to read the private dataset and write the output model repo.
token = os.environ.get("HF_TOKEN")

# Keep every setting overridable from `hf jobs run --env KEY=value`.
dataset_id = os.getenv("DATASET_ID", "tejaskale/trocr-handwriting-lines")
model_id = os.getenv("MODEL_ID", "microsoft/trocr-base-handwritten")
out_repo = os.getenv("OUT_REPO", "tejaskale/trocr-lines-ft")
epochs = int(os.getenv("EPOCHS", "15"))
lr = float(os.getenv("LR", "3e-5"))
batch_size = int(os.getenv("BATCH_SIZE", "8"))
max_length = int(os.getenv("MAX_LENGTH", "128"))
# Regularisation knobs, defaulted for a proper (not overfit-by-design) run over a few
# hundred short line samples: weight decay, a decaying LR schedule, and periodic holdout
# loss every epoch (VAL_LOSS_EVERY=1) so overfitting onset is visible from epoch 1.
weight_decay = float(os.getenv("WEIGHT_DECAY", "0.01"))
lr_schedule = os.getenv("LR_SCHEDULE", "linear")
val_loss_every = int(os.getenv("VAL_LOSS_EVERY", "1"))
# EPOCHS=0 turns this into an inference-only scoring run: no optimiser step, no push of
# the model/processor to the hub, just generation+CER/WER over both splits uploaded to
# OUT_REPO under RESULTS_NAME. Pair with MODEL_ID=<already fine-tuned repo>.
results_name = os.getenv("RESULTS_NAME", "trocr_lines_eval.json")
# Smoke-testing escape hatch: skip pushing the model/processor and results JSON to the
# Hub entirely, e.g. when running a tiny local smoke test with a stubbed dataset. Real
# training/eval runs should leave this at its default (push everything, as train_trocr.py
# always does).
push_to_hub = os.getenv("PUSH_TO_HUB", "1") not in ("0", "false", "False")

# HF Jobs gives us CUDA on GPU flavours; this keeps the script runnable on CPU too.
device = "cuda" if torch.cuda.is_available() else "cpu"

# The dataset was uploaded (src/prepare_line_dataset.py) as a Hugging Face Dataset with
# columns: image (line crop), text (paired label line), page_id, line_index. Unlike
# train_trocr.py, no further splitting is needed here: the dataset already ships fixed
# train/holdout splits at the page level (see src/prepare_line_dataset.py's TRAIN_PAGES /
# HOLDOUT_PAGES), so line-level train items and holdout items are simply each split as-is.
# `load_from_disk` is a local-testing escape hatch: if DATASET_ID points at a directory
# that exists on disk (e.g. a stubbed dataset for a smoke test), load it from there
# instead of hitting the Hub -- production usage (a Hub repo id) is unaffected.
if os.path.isdir(dataset_id):
    ds = load_from_disk(dataset_id)
else:
    ds = load_dataset(dataset_id, token=token)
train_items = list(ds["train"])
holdout_items = list(ds["holdout"])
print({"num_train": len(train_items), "num_holdout": len(holdout_items)})

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

    # Tokenise the line transcripts as labels, fixed to `max_length` tokens. Lines are
    # short (a handful of words to one wrapped sentence), so max_length=128 comfortably
    # covers them with room to spare -- unlike train_trocr.py's 512-token full-page cap.
    labels = processor.tokenizer(
        [x["text"] for x in batch],
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids

    # Padding tokens are filler, not real text. Cross-entropy loss ignores label
    # positions set to -100, so replace pad token ids with -100 before handing labels
    # to the model, exactly as in train_trocr.py.
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": pixel_values, "labels": labels.to(device)}


# Shuffled every epoch: with ~150 short line samples, batch composition variety matters
# more for regularisation than it did for train_trocr.py's ~14-sample full-page batches.
train_loader = DataLoader(train_items, batch_size=batch_size, shuffle=True, collate_fn=collate)
# Fixed batching (no shuffle) for the periodic validation-loss check below, so holdout
# loss is comparable epoch to epoch.
holdout_loader = DataLoader(holdout_items, batch_size=batch_size, shuffle=False, collate_fn=collate)

optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
total_steps = max(1, epochs * len(train_loader))
scheduler = get_scheduler(lr_schedule, optimizer=optim, num_warmup_steps=0, num_training_steps=total_steps)

# EPOCHS=0 means "inference only": skip training entirely and go straight to eval below,
# so an already fine-tuned MODEL_ID can be scored without touching the optimiser.
for epoch in range(epochs):
    model.train()
    train_losses = []
    for batch in train_loader:
        loss = model(**batch).loss
        loss.backward()
        optim.step()
        scheduler.step()
        optim.zero_grad()
        train_losses.append(loss.detach().float().cpu())
    epoch_log = {"epoch": epoch + 1, "train_loss": float(sum(train_losses) / len(train_losses))}

    # Peek at holdout loss every epoch by default (VAL_LOSS_EVERY=1), so overfitting
    # onset on this small, short-sample dataset is visible without changing what the
    # model is trained on.
    if val_loss_every > 0 and (epoch + 1) % val_loss_every == 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in holdout_loader:
                val_losses.append(model(**batch).loss.detach().float().cpu())
        epoch_log["val_loss"] = float(sum(val_losses) / len(val_losses))

    print(epoch_log)


def normalize(value):
    # lowercase, collapse all whitespace (incl. newlines) to single spaces, strip.
    return re.sub(r"\s+", " ", value.lower()).strip()


def generate_predictions(items):
    """Generate a prediction for every item, returned in the same order as `items`."""
    predictions = []
    with torch.no_grad():
        for item in items:
            pixel_values = processor(images=item["image"].convert("RGB"), return_tensors="pt").pixel_values.to(device)
            ids = model.generate(pixel_values, max_length=max_length, num_beams=1, do_sample=False)
            predictions.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
    return predictions


def evaluate_lines(items, predictions, split_name):
    # Line-level CER/WER: each (image, text) pair scored independently. This is the
    # granularity the model is actually trained on.
    results = {}
    hyp_norm_all, ref_norm_all = [], []
    for item, predicted in zip(items, predictions):
        reference = item["text"]
        hyp_norm = normalize(predicted)
        ref_norm = normalize(reference)
        line_cer = jiwer.cer(ref_norm, hyp_norm)
        line_wer = jiwer.wer(ref_norm, hyp_norm)
        key = f"{item['page_id']}_{item['line_index']}"
        print({"split": split_name, "granularity": "line", "line_id": key, "cer": line_cer, "wer": line_wer})

        results[key] = {
            "page_id": item["page_id"],
            "line_index": item["line_index"],
            "predicted": predicted,
            "reference": reference,
            "cer": line_cer,
            "wer": line_wer,
        }
        hyp_norm_all.append(hyp_norm)
        ref_norm_all.append(ref_norm)

    overall_cer = jiwer.cer(ref_norm_all, hyp_norm_all)
    overall_wer = jiwer.wer(ref_norm_all, hyp_norm_all)
    print({"split": split_name, "granularity": "line", "overall_cer": overall_cer, "overall_wer": overall_wer})
    return {"lines": results, "overall_cer": overall_cer, "overall_wer": overall_wer}


def evaluate_pages(items, predictions, split_name):
    # Page-level CER/WER: join predicted lines per page_id (sorted by line_index) with
    # newlines and score against the joined reference lines. This is the granularity
    # that's actually comparable to the earlier full-page (train_trocr.py) results.
    by_page = {}
    for item, predicted in zip(items, predictions):
        by_page.setdefault(item["page_id"], []).append((item["line_index"], item["text"], predicted))

    results = {}
    hyp_norm_all, ref_norm_all = [], []
    for page_id, rows in by_page.items():
        rows.sort(key=lambda r: r[0])
        reference = "\n".join(r[1] for r in rows)
        predicted = "\n".join(r[2] for r in rows)
        hyp_norm = normalize(predicted)
        ref_norm = normalize(reference)
        page_cer = jiwer.cer(ref_norm, hyp_norm)
        page_wer = jiwer.wer(ref_norm, hyp_norm)
        print({"split": split_name, "granularity": "page", "page_id": page_id, "cer": page_cer, "wer": page_wer})

        results[page_id] = {
            "num_lines": len(rows),
            "predicted": predicted,
            "reference": reference,
            "cer": page_cer,
            "wer": page_wer,
        }
        hyp_norm_all.append(hyp_norm)
        ref_norm_all.append(ref_norm)

    overall_cer = jiwer.cer(ref_norm_all, hyp_norm_all)
    overall_wer = jiwer.wer(ref_norm_all, hyp_norm_all)
    print({"split": split_name, "granularity": "page", "overall_cer": overall_cer, "overall_wer": overall_wer})
    return {"pages": results, "overall_cer": overall_cer, "overall_wer": overall_wer}


def evaluate(items, split_name):
    predictions = generate_predictions(items)
    return {
        "line": evaluate_lines(items, predictions, split_name),
        "page": evaluate_pages(items, predictions, split_name),
    }


# Evaluate both splits and both granularities: train_items measures memorisation,
# holdout_items measures actual generalisation to unseen pages/handwriting.
model.eval()
output = {
    "model_id": model_id,
    "dataset_id": dataset_id,
    "num_train": len(train_items),
    "num_holdout": len(holdout_items),
    "train": evaluate(train_items, "train"),
    "holdout": evaluate(holdout_items, "holdout"),
}
results_path = results_name
with open(results_path, "w") as f:
    json.dump(output, f, indent=2)

# EPOCHS=0 is an inference-only scoring run: only the results JSON is uploaded, so the
# model/processor already living at OUT_REPO (or wherever MODEL_ID pointed) is untouched.
# PUSH_TO_HUB=0 additionally skips uploading anything at all, for local smoke testing.
if push_to_hub:
    if epochs > 0:
        # Save the fine-tuned model and processor together so inference can load one
        # repo, and push the CER/WER results alongside them so the numbers travel with
        # the model.
        model.push_to_hub(out_repo, private=True, token=token)
        processor.push_to_hub(out_repo, private=True, token=token)
    HfApi().upload_file(
        path_or_fileobj=results_path,
        path_in_repo=results_path,
        repo_id=out_repo,
        token=token,
    )
