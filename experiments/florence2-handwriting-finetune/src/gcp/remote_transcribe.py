"""Runs on the GCP GPU VM: transcribes handwriting images with a given model.

This is the compute stage of the pipeline. run_gcp_pipeline.sh scps this
file, models.json, and requirements-remote.txt up to a freshly provisioned
L4 GPU VM, then invokes it once per model listed in models.json (Florence-2
first, then the Qwen/Gemma chat-style vision models). For each model it
walks every sanitised image under --input and writes a <stem>.txt (the
cleaned transcription) and a <stem>.json (run metadata: timing, status,
attempt count) into --output/<model-slug>/, so results from different
models never collide and can be compared side by side.

Multi-prompt design: each image is retried with up to --attempts different
prompts (see PROMPTS below) until one produces non-empty text.
  + Pro: more robust to a model refusing/echoing/misreading one particular
    prompt phrasing, and lets us compare how sensitive a model is to prompt
    wording (useful when picking a model for the real workflow).
  - Con: worst case (model returns empty every time) costs up to 3x the
    inference of a single-prompt approach, and even the common case pays
    for the *first* prompt's generation before knowing whether the *last*
    prompt would have worked better - there's no way to pick the "best" of
    the three, only the first that returns anything.
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
# Human-readable, filesystem-safe output directory names per model, so
# --output/<slug>/ is stable and legible even though HF model IDs contain
# slashes and dots. Any model_id not listed here falls back to a generated
# slug (see model_slug below) rather than failing.
MODEL_SLUGS = {
    "microsoft/Florence-2-base": "florence-2-base",
    "Qwen/Qwen3.5-0.8B": "qwen3.5-0.8b",
    "Qwen/Qwen3.5-2B": "qwen3.5-2b",
    "Qwen/Qwen3.5-4B": "qwen3.5-4b",
    "google/gemma-4-E2B-it": "gemma-4-e2b",
    "google/gemma-4-E4B-it": "gemma-4-e4b",
    "Qwen/Qwen3.5-9B": "qwen3.5-9b",
}
# Three worded differently on purpose: run_image (below) tries them in
# order, one per attempt, stopping at the first prompt that yields
# non-empty cleaned text. This is the multi-prompt retry strategy described
# in the module docstring - it trades extra inference cost for resilience
# against a single prompt phrasing that a given model handles poorly.
PROMPTS = [
    "Transcribe the handwriting in this image exactly. Return only the transcription text.",
    "Read the handwritten English text. Preserve line breaks. Return no commentary, labels, markdown, or quotes.",
    "OCR task: output only the visible handwritten text, in reading order. If unsure, write the closest literal transcription.",
]


def model_slug(model):
    """Look up (or derive) the filesystem-safe output folder name for a model."""
    return MODEL_SLUGS.get(model, re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-"))


def image_paths(root):
    """List supported image files directly under root, sorted for a stable run order."""
    return sorted(p for p in Path(root).expanduser().iterdir() if p.suffix.lower() in EXTS)


def clean_text(text):
    """Strip formatting/preamble that chat-style vision models tend to add.

    Unlike Florence-2's structured <OCR> task output, the Qwen/Gemma chat
    backends are instruction-following models: even when asked for "only
    the transcription text", they sometimes wrap the answer in a markdown
    code fence (```text ... ```) or prefix it with a label like
    "Transcription:" or "Output:". Those artifacts would otherwise end up
    baked into the .txt transcript, so they're stripped heuristically here
    rather than relying on prompting alone to prevent them.
    """
    text = text.strip().replace("```text", "").replace("```", "").strip()
    prefixes = ["transcription:", "text:", "output:"]
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


# --- Backends -----------------------------------------------------------
# Two backends because the models this pipeline runs use two different
# transformers APIs: Florence-2 is a task-token captioning/OCR model
# (invoked with a fixed "<OCR>" pseudo-prompt and its own post-processing),
# while the Qwen/Gemma models are general chat/instruction vision-language
# models (invoked via a chat template with a free-text prompt). load_backend
# below picks the right one per model_id.
class FlorenceBackend:
    def __init__(self, model_id, load_in_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        kwargs = {"trust_remote_code": True, "attn_implementation": "eager", "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not load_in_4bit:
            self.model = self.model.to(self.device)

    def transcribe(self, path, prompt, max_new_tokens):
        # NOTE: `prompt` is accepted for interface parity with ChatBackend
        # (see run_image, which passes a different PROMPTS entry per
        # attempt) but is intentionally ignored - Florence-2 is driven by
        # its fixed "<OCR>" task token, not free-text instructions, so all
        # 3 attempts for this model use an identical prompt. Its retries
        # only help if generation is non-deterministic or transiently fails.
        image = Image.open(path).convert("RGB")
        inputs = self.processor(text="<OCR>", images=image, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(next(self.model.parameters()).dtype)
        generated = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=False)
        text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task="<OCR>", image_size=(image.width, image.height))
        return parsed.get("<OCR>", text)


class ChatBackend:
    def __init__(self, model_id, load_in_4bit=False):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=1600 * 1600)

    def transcribe(self, path, prompt, max_new_tokens):
        # Unlike FlorenceBackend, `prompt` is actually used here: each of
        # the 3 PROMPTS is rendered through the model's own chat template
        # (enable_thinking=False skips Qwen's chain-of-thought preamble,
        # which we don't want mixed into a plain-text transcript).
        image = Image.open(path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        # Chat models echo the prompt back at the start of `generated`;
        # slicing off the input length keeps only the newly generated
        # continuation (the answer), not the prompt we just fed in.
        generated = generated[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def load_backend(model, load_in_4bit=False):
    """Pick FlorenceBackend for the Florence OCR model, ChatBackend otherwise."""
    if model == "microsoft/Florence-2-base":
        return FlorenceBackend(model, load_in_4bit)
    return ChatBackend(model, load_in_4bit)


def write_record(outdir, path, model, attempt_count, status, text, runtime):
    """Write <stem>.txt (the transcript) and <stem>.json (run metadata).

    Two files per image rather than one combined file so the .txt can be
    consumed directly (e.g. diffed, grepped, fed to another tool) without
    parsing JSON, while the .json still captures how the result was
    produced (status/attempt_count/timing) for auditing/debugging. Note:
    a .txt is written even on "parse-failed" (empty text) - see main(),
    which uses the *existence* of both files as its "already done" check,
    so a failed image is not retried by re-running this script as-is.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{path.stem}.txt").write_text(text.rstrip() + ("\n" if text else ""))
    data = {
        "model_id": model,
        "runtime_seconds": round(runtime, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attempt_count": attempt_count,
        "status": status,
    }
    (outdir / f"{path.stem}.json").write_text(json.dumps(data, indent=2) + "\n")


def run_image(backend, path, max_new_tokens, attempts):
    """Transcribe one image, retrying with the next PROMPTS entry on empty output.

    Stops at the first attempt that yields non-empty cleaned text ("ok").
    If every attempt (up to `attempts`, capped to len(PROMPTS) - later
    attempts reuse the last prompt) comes back empty, returns whatever the
    last attempt produced (empty string) with status "parse-failed" rather
    than raising, so one bad image doesn't abort the whole model run.
    """
    start = time.perf_counter()
    text = ""
    attempt_count = 0
    for i in range(attempts):
        attempt_count = i + 1
        text = clean_text(backend.transcribe(path, PROMPTS[min(i, len(PROMPTS) - 1)], max_new_tokens))
        if text:
            return text, attempt_count, "ok", time.perf_counter() - start
        print(path.stem, "parse-failed", attempt_count, "empty response", flush=True)
    return text, attempt_count, "parse-failed", time.perf_counter() - start


def main(argv=None):
    """Entry point: load one model, transcribe every image under --input.

    Invoked once per model_id by run_gcp_pipeline.sh's remote ssh command
    (see models.json for the list of models/settings). Skips any image
    whose .txt and .json already exist in this model's output folder, which
    makes a re-run of the same model resumable after a crash/interruption -
    but note this also means an image that finished as "parse-failed" will
    NOT be retried by simply re-running, since both files still get written
    for a failed attempt (see write_record).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", default=512, type=int)
    parser.add_argument("--attempts", default=3, type=int)
    args = parser.parse_args(argv)
    backend = load_backend(args.model, args.load_in_4bit)
    outdir = Path(args.output).expanduser() / model_slug(args.model)
    for path in image_paths(args.input):
        txt = outdir / f"{path.stem}.txt"
        js = outdir / f"{path.stem}.json"
        if txt.exists() and js.exists():
            print(path.stem, "exists", flush=True)
            continue
        text, attempt_count, status, runtime = run_image(backend, path, args.max_new_tokens, args.attempts)
        write_record(outdir, path, args.model, attempt_count, status, text, runtime)
        print(path.stem, status, flush=True)


if __name__ == "__main__":
    main()
