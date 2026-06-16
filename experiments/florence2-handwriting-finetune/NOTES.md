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
