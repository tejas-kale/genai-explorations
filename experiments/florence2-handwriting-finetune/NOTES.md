# Florence-2 handwriting OCR notes

## Goal

Fine-tune `microsoft/Florence-2-base-ft` for basic OCR on a tiny labelled handwriting dataset: about 30 image/transcript pairs. Keep the implementation naive, inspectable, and runnable on cheap GPU hardware.

## Data layout

Original local data:

```text
experiments/florence2-handwriting-finetune/
  labels/              # one .txt transcript per labelled image
  upload_sanitised/    # JPEG images
  data/                # original HEIC images and generated train copy, ignored by git
```

A labelled image is matched by stem:

```text
labels/IMG_5472.txt
upload_sanitised/IMG_5472.jpg
```

We created `data/train/images` and `data/train/labels` as a local copied training layout, but avoid committing private data. The useful long-term route is the private Hugging Face dataset.

## Hugging Face dataset

Dataset repo:

```text
tejaskale/florence-handwriting-ocr
```

It is private and contains a Hugging Face Dataset with:

```text
train:
  image: Image()
  text: string
  image_id: string

infer:
  image: Image()
  text: ""
  image_id: string
```

The `infer` split is for an unlabelled image, currently the first JPEG in `upload_sanitised` without a matching label, e.g. `IMG_5502`.

Uploader:

```bash
HF_TOKEN=... python experiments/florence2-handwriting-finetune/src/hf/upload_dataset.py
```

The token must be able to write the dataset repo.

## Why training moved off Colab

Colab repeatedly hit Florence/Transformers remote-code issues and notebook state problems. HF Jobs is cleaner because:

- the private HF dataset is loaded directly;
- the training environment is reproducible from a small shell script;
- the fine-tuned model can be pushed straight back to HF;
- no manual upload into Colab is needed.

## Training script

Main script:

```text
src/hf/train_florence.py
```

Run script used by HF Jobs:

```text
scripts/run_hf_training.sh
```

Output model repo defaults to:

```text
tejaskale/florence-handwriting-ocr-ft
```

Important defaults:

```text
MODEL_ID=microsoft/Florence-2-base-ft
FLORENCE_REVISION=refs/pr/6
EPOCHS=10
LR=1e-6
OUT_REPO=tejaskale/florence-handwriting-ocr-ft
```

Run HF training job:

```bash
hf jobs run --flavor t4-small --secrets HF_TOKEN --env OUT_REPO=tejaskale/florence-handwriting-ocr-ft python:3.11 bash -c 'curl -fsSL https://raw.githubusercontent.com/tejas-kale/genai-explorations/main/experiments/florence2-handwriting-finetune/scripts/run_hf_training.sh | bash'
```

If local `HF_TOKEN` is invalid, the HF CLI may fail before submitting. Use:

```bash
unset HF_TOKEN
hf auth login
```

Then rerun the job. `--secrets HF_TOKEN` passes the logged-in HF token into the job.

## Dependency/version lessons

Florence-2 uses `trust_remote_code=True`. That means model code is downloaded from the model repo, and small version mismatches can break runtime behaviour.

Known fixes used here:

1. Pin Florence revision:

```python
revision = "refs/pr/6"
```

2. Pin Transformers in the HF job script:

```bash
pip install -U 'transformers==4.41.2' datasets accelerate timm einops pillow huggingface_hub torch torchvision
```

3. Force eager attention:

```python
attn_implementation="eager"
```

Reason: faster attention implementations caused compatibility errors with Florence custom code. Eager is slower but stable enough for 30 images.

## Generation bugs encountered

Training loss computation worked before generation did. The smoke-test generation exposed multiple issues.

### Missing `generate()` on inner language model

Error:

```text
AttributeError: 'Florence2LanguageForConditionalGeneration' object has no attribute 'generate'
```

Fix:

```python
from transformers.generation import GenerationMixin
model.language_model.__class__ = type("Florence2LanguageModel", (model.language_model.__class__, GenerationMixin), {})
```

Florence's outer model calls `generate()` on an internal language model. Newer Transformers expects that inner model to inherit `GenerationMixin`. This line keeps the original class and adds the missing generation helper methods.

### Missing generation config

Error:

```text
AttributeError: 'NoneType' object has no attribute '_from_model_config'
```

Fix:

```python
from transformers import GenerationConfig
model.language_model.generation_config = GenerationConfig.from_model_config(model.language_model.config)
```

### Beam search/cache bug

Error:

```text
AttributeError: 'NoneType' object has no attribute 'shape'
```

Fix: use greedy generation and disable cache:

```python
model.generate(..., num_beams=1, do_sample=False, use_cache=False)
```

This is less fancy but enough for a smoke test.

## Training behaviour seen

With 3 epochs, one run printed roughly:

```text
epoch 1 train_loss ~3.4 val_loss ~1.9
epoch 2 train_loss ~3.0 val_loss ~1.75
epoch 3 train_loss ~3.0 val_loss ~1.71
```

Loss going down means the model is fitting the transcript tokens better. This is only a sanity check, not proof of OCR quality.

## Validation loss is not OCR quality

Validation loss measures whether the model assigns higher probability to the correct transcript tokens on held-out examples. It does not directly tell us:

- character error rate;
- word error rate;
- whether line breaks are acceptable;
- whether the output is useful to a human.

For real OCR evaluation, add CER/WER on generated text. We skipped that deliberately to keep the first version tiny.

## Vision tower freezing

Florence has an image-reading part, called the vision tower. It turns pixels into internal visual features. We freeze it:

```python
for p in model.vision_tower.parameters():
    p.requires_grad = False
```

Why freeze:

- uses less GPU memory;
- trains faster;
- less likely to overfit 30 images;
- less likely to OOM on T4.

Trade-off:

- if handwriting/image style is far from Florence's pretraining, the image reader cannot adapt;
- unfreezing might improve OCR but costs more memory and can overfit.

Lazy default: keep frozen first. Unfreeze only after seeing poor generated OCR.

## Learning rate notes

Current default:

```text
LR=1e-6
```

How to play:

- loss barely moves: try `3e-6` or `1e-5`;
- loss jumps around or output degrades: try `3e-7`;
- change one thing at a time: LR, then epochs, then vision tower freezing.

Scheduler:

```python
sched = get_scheduler("linear", optim, 0, epochs * len(train_loader))
```

This starts at `lr` and lowers it linearly to zero across training. Early updates are larger; later ones are gentler.

## Inference script

Inference script:

```text
src/hf/infer_florence.py
```

HF job script:

```text
scripts/run_hf_inference.sh
```

Run after dataset has an `infer` split and trained model exists:

```bash
hf jobs run --flavor t4-small --secrets HF_TOKEN python:3.11 bash -c 'curl -fsSL https://raw.githubusercontent.com/tejas-kale/genai-explorations/main/experiments/florence2-handwriting-finetune/scripts/run_hf_inference.sh | bash'
```

Expected output shape:

```python
{"image_id": "IMG_5502", "predicted": "..."}
```

## Shell/HF Jobs gotchas

HF Jobs syntax is:

```bash
hf jobs run [OPTIONS] IMAGE COMMAND...
```

The image must come before the command. The correct secret flag is `--secrets`, not `--secret`.

Working pattern:

```bash
hf jobs run --flavor t4-small --secrets HF_TOKEN python:3.11 bash -c 'curl -fsSL URL | bash'
```

Avoid multi-line quoted commands in zsh. Newlines inside quotes caused HF Jobs to treat the entire command as a missing executable.

## Git/data hygiene

Private/local data is ignored:

```text
experiments/florence2-handwriting-finetune/.ui/
experiments/florence2-handwriting-finetune/labels/
experiments/florence2-handwriting-finetune/upload_sanitised/
experiments/florence2-handwriting-finetune/experiments/*
```

Do not commit HF tokens. One notebook version accidentally contained a hard-coded token; it was removed and the notebook was later deleted.

## Current philosophy

Keep this deliberately naive:

- no Trainer;
- no LoRA;
- no checkpointing;
- no metrics zoo;
- batch size 1;
- one small script that can be read top to bottom.

Add complexity only after confirming the naive version produces useful OCR.

## TrOCR experiments (July 2026)

A parallel track fine-tuned `microsoft/trocr-base-handwritten` on the same data, reusing the Florence HF Jobs setup. TrOCR is a stock `VisionEncoderDecoderModel` — none of Florence's remote-code workarounds apply.

Scripts:

```text
src/hf/train_trocr.py           # full-page training, 70/30 page split, both-split CER/WER eval
src/hf/train_trocr_lines.py     # line-level variant
src/prepare_line_dataset.py     # builds line-crop dataset from pages + labels
scripts/run_trocr_training.sh
scripts/run_trocr_lines_training.sh
scripts/trocr_eval.py           # local zero-shot pipeline (segmentation + eval)
```

Model/dataset repos: `tejaskale/trocr-handwriting-ocr-ft` (overfit), `tejaskale/trocr-handwriting-ocr-reg` (regularised), `tejaskale/trocr-lines-ft` (line-level), `tejaskale/trocr-handwriting-lines` (line dataset, 156 train / 71 holdout crops).

### Results

All CER/WER after lowercasing and whitespace collapsing. Page split: 21 train / 9 holdout pages, seed 42.

```text
run                                   train CER   holdout CER
zero-shot line pipeline (local, M1)   -           ~0.74
full-page overfit (100 ep, LR 5e-5)   0.031       0.82
full-page regularised (20 ep)         1.44        1.29
line-level regularised (15 ep)        0.91        0.98
```

No fine-tuned variant beat zero-shot on unseen pages.

### Lessons

1. **Full-page input is out of distribution for TrOCR.** The encoder resizes everything to 384x384; a full page makes handwriting a few pixels tall. The overfit run proved the pipeline can memorise (18/21 training pages reproduced with CER 0.0) but holdout CER 0.82 shows it learned page-layout-to-transcript lookup, not reading.

2. **Memorisation is a cheap, useful pipeline smoke test.** 100 epochs, constant LR 5e-5, no weight decay, batch 2. Train loss 9.4 → 0.02. If a model cannot even memorise ~20 pages, something is broken upstream.

3. **The labels are logical lines, not physical lines.** Each label line is a sentence/bullet that wraps across 2-7 physical ink rows. Naive count-matching of segmented rows to label lines silently misaligns pairs. The workaround — allocate rows to label lines proportionally by character length and stack them into composite crops — produced multi-row images that are nearly as out-of-distribution as full pages, plus alignment noise. Result: even train-split generation was garbage (CER 0.91) while train loss fell to 1.3. Teacher-forced loss can look healthy while free-running generation fails.

4. **Val loss called the overfit early.** In the line run, holdout loss bottomed at epoch 3-6 and rose after; per-epoch val loss (`VAL_LOSS_EVERY=1`) is cheap and worth always logging.

5. **Pin transformers in job scripts.** Unpinned `pip install -U transformers` pulled 5.13, which fails to load TrOCR's tokenizer without `sentencepiece`. Pinned to 5.10.2 (the locally smoke-tested version) + `sentencepiece`.

6. **Smoke-test locally before submitting jobs.** A one-forward-pass + one-generate check on CPU with 2 local examples caught a `model.config.max_length` ValueError (generation knobs must go on `model.generation_config` in current transformers) before it could burn a job.

### Where this leaves TrOCR

Fine-tuning TrOCR on this dataset as currently labelled is a dead end. Viable continuations:

- Re-label a subset at physical-line granularity (each ink row gets its own text); even ~10 pages would yield ~200 clean single-row pairs, which is what TrOCR actually consumes.
- Otherwise prefer the VLM track (Florence/Qwen/Gemma), which handles full pages with logical-line transcripts natively.

A physical-line re-labelling UI was built, but its projection-profile segmentation produced imprecise crops (merged lines, cut lines, half lines) — unusable as ground truth without better line detection (e.g. Apple Vision bounding boxes). Work stopped here.

## Decision: no more fine-tuning (July 2026)

Fine-tuning is abandoned. With ~30 labelled pages, every fine-tuned variant (Florence full-page, TrOCR full-page, TrOCR line-level) lost to alternatives that need no training, and clean line-level ground truth would require re-labelling effort disproportionate to the goal.

The intention of this project is to scan handwritten notes for transfer to other apps (Ulysses, Bear, Lettera, Journal). The workflow is:

1. As needed, scan a bunch of pages.
2. Convert the pages to text with the Qwen 9B model via the existing GCP VM pipeline (`scripts/run_gcp_pipeline.sh` + `src/gcp/remote_transcribe.py`, output per model under `experiments/`).
3. Build further workflows to route the returned text into the target apps (to be tackled separately).

All fine-tuning code was removed; the GCP pipeline, `prepare_upload.py`, `models.json`, and the remote requirements were kept and annotated. The TrOCR HF artifacts (models `trocr-handwriting-ocr-ft`, `trocr-handwriting-ocr-reg`, `trocr-lines-ft`; dataset `trocr-handwriting-lines`) were deleted; the Florence dataset and model repos remain.
