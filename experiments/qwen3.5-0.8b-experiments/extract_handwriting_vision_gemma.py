import subprocess
import time
from pathlib import Path

import requests

MODEL = "gemma4:e2b"
IMAGE = Path.home() / "Downloads" / "IMG_5468.HEIC"
URL = "http://localhost:11434"
SWIFT = Path("vision_ocr.swift")
OUT_RAW = Path("IMG_5468_vision_raw.txt")
OUT_TXT = Path("IMG_5468_transcript.txt")

SWIFT.write_text(r'''
import Foundation
import Vision
import AppKit

let path = CommandLine.arguments[1]
let url = URL(fileURLWithPath: path)
guard let image = NSImage(contentsOf: url) else { fatalError("cannot read image") }
guard let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { fatalError("cannot make cgimage") }

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = ["en-US"]
request.minimumTextHeight = 0.005

let handler = VNImageRequestHandler(cgImage: cg, options: [:])
try handler.perform([request])
for observation in request.results ?? [] {
    if let candidate = observation.topCandidates(1).first {
        print(candidate.string)
    }
}
''')


def running():
    try:
        return requests.get(f"{URL}/api/tags", timeout=2).ok
    except Exception:
        return False


def dims(path):
    out = subprocess.check_output(["magick", "identify", "-format", "%w %h", path], text=True)
    return tuple(map(int, out.split()))


def ocr(path):
    return subprocess.run(["swift", str(SWIFT), path], check=True, capture_output=True, text=True).stdout.strip()


def make_base_images():
    commands = {
        "plain.png": ["magick", str(IMAGE), "plain.png"],
        "deskew.png": ["magick", str(IMAGE), "-resize", "220%", "-colorspace", "Gray", "-auto-level", "-deskew", "40%", "deskew.png"],
        "contrast.png": ["magick", str(IMAGE), "-resize", "260%", "-colorspace", "Gray", "-auto-level", "-contrast-stretch", "1%x1%", "-sharpen", "0x1", "contrast.png"],
        "threshold.png": ["magick", str(IMAGE), "-resize", "260%", "-colorspace", "Gray", "-auto-level", "-threshold", "58%", "threshold.png"],
    }
    for command in commands.values():
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return list(commands)


def make_strips(path, n=14, overlap=0.35):
    w, h = dims(path)
    band = int(h / n)
    step = int(band * (1 - overlap))
    crops = []
    y = 0
    i = 0
    while y < h:
        ch = min(int(band * (1 + overlap)), h - y)
        out = f"{Path(path).stem}_strip_{i:02d}.png"
        subprocess.run(["magick", path, "-crop", f"{w}x{ch}+0+{y}", "+repage", "-resize", "160%", out], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        crops.append(out)
        y += step
        i += 1
    return crops


def collect_ocr():
    chunks = []
    for base in make_base_images():
        chunks.append(f"--- FULL {base} ---\n{ocr(base)}")
        for strip in make_strips(base):
            text = ocr(strip)
            if text:
                chunks.append(f"--- STRIP {strip} ---\n{text}")
    raw = "\n\n".join(chunks)
    OUT_RAW.write_text(raw)
    return raw


def clean_with_gemma(raw):
    prompt = """You are cleaning OCR from cursive handwriting.
Use the repeated crop outputs to recover missing lines.
Preserve likely wording. Do not invent details.
Use [unclear] for unreadable words.
Keep line breaks in reading order.
Return only the transcript.

OCR outputs:
""" + raw[-18000:]
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}, "keep_alive": 0}
    response = requests.post(f"{URL}/api/generate", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()["response"].strip()


started = not running()
proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if started else None
try:
    for _ in range(60):
        if running():
            break
        time.sleep(0.5)
    raw = collect_ocr()
    transcript = clean_with_gemma(raw)
    OUT_TXT.write_text(transcript)
    print(transcript)
    print(f"\nRAW_OCR={OUT_RAW}")
    print(f"TRANSCRIPT={OUT_TXT}")
finally:
    try:
        requests.post(f"{URL}/api/generate", json={"model": MODEL, "keep_alive": 0}, timeout=10)
    except Exception:
        pass
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
