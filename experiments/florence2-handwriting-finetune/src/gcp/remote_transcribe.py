import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
MODEL_SLUGS = {
    "microsoft/Florence-2-base": "florence-2-base",
    "Qwen/Qwen3.5-0.8B": "qwen3.5-0.8b",
    "Qwen/Qwen3.5-2B": "qwen3.5-2b",
    "Qwen/Qwen3.5-4B": "qwen3.5-4b",
    "google/gemma-4-E2B-it": "gemma-4-e2b",
    "google/gemma-4-E4B-it": "gemma-4-e4b",
    "Qwen/Qwen3.5-9B": "qwen3.5-9b",
}
PROMPTS = [
    "Transcribe the handwriting in this image exactly. Return only the transcription text.",
    "Read the handwritten English text. Preserve line breaks. Return no commentary, labels, markdown, or quotes.",
    "OCR task: output only the visible handwritten text, in reading order. If unsure, write the closest literal transcription.",
]


def model_slug(model):
    return MODEL_SLUGS.get(model, re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-"))


def image_paths(root):
    return sorted(p for p in Path(root).expanduser().iterdir() if p.suffix.lower() in EXTS)


def clean_text(text):
    text = text.strip().replace("```text", "").replace("```", "").strip()
    prefixes = ["transcription:", "text:", "output:"]
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


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
        image = Image.open(path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def load_backend(model, load_in_4bit=False):
    if model == "microsoft/Florence-2-base":
        return FlorenceBackend(model, load_in_4bit)
    return ChatBackend(model, load_in_4bit)


def write_record(outdir, path, model, attempt_count, status, text, runtime):
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
