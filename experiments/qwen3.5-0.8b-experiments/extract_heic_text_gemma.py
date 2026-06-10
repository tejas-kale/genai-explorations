import base64
import subprocess
import time
from pathlib import Path

import requests

MODEL = "gemma4:e2b"
IMAGE = Path.home() / "Downloads" / "IMG_5468.HEIC"
PNG = Path("IMG_5468_gemma.png")
URL = "http://localhost:11434"


def running():
    return requests.get(f"{URL}/api/tags", timeout=2).ok


started = not running()
proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if started else None
try:
    for _ in range(60):
        if running():
            break
        time.sleep(0.5)

    subprocess.run(["sips", "-s", "format", "png", str(IMAGE), "--out", str(PNG)], check=True, stdout=subprocess.DEVNULL)
    image = base64.b64encode(PNG.read_bytes()).decode()
    prompt = "Extract all visible text from this image as faithfully as possible. Preserve line breaks. Do not describe the image. If text is unclear, mark it as [unclear]."
    payload = {"model": MODEL, "prompt": prompt, "images": [image], "stream": False, "options": {"temperature": 0}, "keep_alive": 0}
    response = requests.post(f"{URL}/api/generate", json=payload, timeout=600)
    response.raise_for_status()
    print(response.json()["response"].strip())
finally:
    requests.post(f"{URL}/api/generate", json={"model": MODEL, "keep_alive": 0}, timeout=10)
    if proc:
        proc.terminate()
        proc.wait(timeout=10)
