# Handwriting transcription baseline pipeline

This directory contains a short-lived GCP VM workflow for generating draft handwriting transcriptions, plus a local evaluator for hand-corrected labels.

## Layout

- `data/` — private source images, not created here.
- `upload_sanitised/` — EXIF-stripped bounded JPEGs made before VM upload.
- `experiments/<model_slug>/<image_stem>.txt` — raw transcription.
- `experiments/<model_slug>/<image_stem>.json` — model id, runtime seconds, timestamp, attempt count, status.
- `labels/<image_stem>.txt` — your corrected ground truth.
- `src/` — local Python tools.
- `src/gcp/` — Python run on the GCP VM.
- `scripts/` — shell wrappers only.

## Models

`models.json` runs, in order:

1. `microsoft/Florence-2-base`
2. `Qwen/Qwen3.5-0.8B`
3. `Qwen/Qwen3.5-2B`
4. `Qwen/Qwen3.5-4B`
5. `google/gemma-4-E2B-it`
6. `google/gemma-4-E4B-it`
7. `Qwen/Qwen3.5-9B` with `--load-in-4bit`
8. Local-only `apple/Vision-VNRecognizeTextRequest`, written under `experiments/apple-vision-ocr/`

## Local Apple Vision OCR

Run this on macOS after installing local requirements:

```bash
python src/apple_vision_ocr.py --input data --output experiments
```

This uses Apple's native Vision OCR and writes the same `.txt`/`.json` layout as the VM models, so local evaluation treats it as another model. It is not part of the GCP VM run.

## Local sanitisation only

```bash
python3 -m venv .venv-local
. .venv-local/bin/activate
pip install -r requirements-local.txt
python src/prepare_upload.py --input data --output upload_sanitised
```

## Full GCP run

The GCP wrapper follows the photo-archivist security shape: custom VPC, no public IP, SSH/SCP via IAP, Cloud NAT for outbound Hugging Face downloads, OS Login, Shielded VM, CMEK-encrypted boot disk, minimal service account, and explicit cleanup.

```bash
export PROJECT=your-gcp-project
export RUN_ID=hw-baseline-001
export REGION=europe-west4
export ZONE=europe-west4-a
./scripts/run_gcp_pipeline.sh
./scripts/cleanup_gcp_pipeline.sh
```

Set `MACHINE`, `ACCELERATOR`, `BOOT_DISK_GB`, `RUN_ID`, `DATA_DIR`, `OUTPUT`, or `SSH_KEY_FILE` if needed. The script copies results back into `experiments/` before cleanup. If `europe-west4-*` has no L4 capacity, `REGION=us-central1 ZONE=us-central1-a` worked for the two-image smoke run.

## Remote script manual use

On the VM:

```bash
python remote_transcribe.py \
  --input input \
  --output experiments \
  --model Qwen/Qwen3.5-9B \
  --load-in-4bit \
  --max-new-tokens 512 \
  --attempts 3
```

Florence-2 uses `<OCR>`. Other models use progressively stricter text-only prompts. Empty responses are retried; exhausted retries write `status: parse-failed` instead of aborting the batch. Florence currently needs `transformers==4.57.3`; the GCP wrapper upgrades Transformers afterwards for the Qwen/Gemma models.

## Label editing UI

After the 9B draft exists, start the local editor:

```bash
python src/label_ui.py --images data --experiments experiments --labels labels --draft-model qwen3.5-9b
```

Open `http://127.0.0.1:8765`. Each page shows the source image beside an editable transcript. It loads `labels/<image_stem>.txt` if present, otherwise the `qwen3.5-9b` draft, and saves edits to `labels/`.

## Local evaluation

After editing drafts into `labels/<image_stem>.txt`:

```bash
. .venv-local/bin/activate
python src/evaluate_handwriting.py --experiments experiments --labels labels
```

The evaluator scores only images with labels present, using `jiwer` CER/WER, and prints the model ranking sorted best to worst.

## Tests

```bash
uv run --with pytest --with pillow --with pillow-heif --with jiwer pytest -q
```
