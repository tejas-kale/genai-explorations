import re
import subprocess
from pathlib import Path
from mlx_lm import load, generate

MODEL_ID = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
IMAGE = Path.home() / "Downloads" / "IMG_5468.HEIC"
VARIANTS = [
    ("plain", ["magick", str(IMAGE), "plain.png"]),
    ("large", ["magick", str(IMAGE), "-resize", "200%", "-colorspace", "Gray", "-sharpen", "0x1", "large.png"]),
    ("threshold", ["magick", str(IMAGE), "-resize", "200%", "-colorspace", "Gray", "-auto-level", "-threshold", "60%", "threshold.png"]),
]

def ocr(path, psm):
    return subprocess.run(["tesseract", path, "stdout", "--psm", str(psm), "-l", "eng"], check=True, capture_output=True, text=True).stdout.strip()

def score(text):
    return len(re.findall(r"[A-Za-z]{3,}", text))

runs = []
for name, command in VARIANTS:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for psm in [3, 4, 6, 11, 12]:
        text = ocr(command[-1], psm)
        runs.append((score(text), name, psm, text))

_, name, psm, raw = max(runs, key=lambda x: x[0])
model, tokenizer = load(MODEL_ID)
messages = [
    {"role": "system", "content": "You are transcribing OCR output. Preserve text faithfully. Fix obvious spacing only. If the OCR is mostly noise, say so briefly, then return the raw readable fragments."},
    {"role": "user", "content": f"OCR variant: {name}, psm={psm}\n\nRaw OCR:\n{raw}\n\nReturn only the most faithful transcription."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
text = generate(model, tokenizer, prompt=prompt, max_tokens=1024)
print(f"BEST OCR VARIANT: {name}, psm={psm}")
print("\nRAW OCR\n=======")
print(raw)
print("\nQWEN TRANSCRIPT\n===============")
print(text.strip())
