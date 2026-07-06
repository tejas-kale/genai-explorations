# Handwriting transcription pipeline

Scans of handwritten notes are transcribed on a short-lived GCP GPU VM so the text can be routed into other apps (Ulysses, Bear, Lettera, Journal). Fine-tuning was explored and abandoned; see `data/NOTES.md` for that history and the current decision.

## Layout

- `data/` — everything that is not code (private, git-ignored except `NOTES.md`):
  - `data/scans/` — original HEIC photos of handwritten pages.
  - `data/labels/<image_stem>.txt` — hand-corrected ground-truth transcripts.
  - `data/upload_sanitised/` — EXIF-stripped bounded JPEGs generated before VM upload.
  - `data/outputs/<model_slug>/<image_stem>.txt|.json` — per-model transcriptions and run metadata.
  - `data/NOTES.md` — experiment notes, lessons, and decisions.
- `src/prepare_upload.py` — local image sanitisation.
- `src/gcp/remote_transcribe.py` — runs on the GCP VM.
- `scripts/` — GCP provisioning/teardown wrappers.
- `models.json` — model roster the VM runs (see below).

## Workflow

1. Drop new page photos into `data/scans/`.
2. Run the GCP pipeline (below). It sanitises images, provisions an isolated VM, transcribes, and copies results into `data/outputs/`.
3. Route the returned text into target apps (workflow to be built).

## Models

`models.json` lists the candidate models; `run_gcp_pipeline.sh` runs every entry except the first (Florence needs an older Transformers, so it runs separately first). `Qwen/Qwen3.5-9B` (4-bit) was the quality winner and is the model the workflow depends on.

## Local sanitisation only

```bash
pip install -r requirements-local.txt
python src/prepare_upload.py --input data/scans --output data/upload_sanitised
```

## Full GCP run

The wrapper follows the photo-archivist security shape: custom VPC, no public IP, SSH/SCP via IAP, Cloud NAT for outbound Hugging Face downloads, OS Login, Shielded VM, CMEK-encrypted boot disk, minimal service account, and explicit cleanup. Both scripts carry detailed comments.

```bash
export PROJECT=your-gcp-project
export RUN_ID=hw-baseline-001
export REGION=europe-west4
export ZONE=europe-west4-a
./scripts/run_gcp_pipeline.sh
./scripts/cleanup_gcp_pipeline.sh
```

Set `MACHINE`, `ACCELERATOR`, `BOOT_DISK_GB`, `RUN_ID`, `DATA_DIR`, `OUTPUT`, or `SSH_KEY_FILE` if needed. If `europe-west4-*` has no L4 capacity, `REGION=us-central1 ZONE=us-central1-a` worked for the two-image smoke run.

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

Florence-2 uses `<OCR>`. Other models use progressively stricter text-only prompts. Empty responses are retried; exhausted retries write `status: parse-failed` instead of aborting the batch. Note the resume logic skips any image whose output files exist, even `parse-failed` ones — delete the image's `.txt`/`.json` pair to force a retry.
