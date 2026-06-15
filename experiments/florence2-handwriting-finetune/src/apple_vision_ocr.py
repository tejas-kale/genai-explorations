import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
MODEL_ID = "apple/Vision-VNRecognizeTextRequest"
SLUG = "apple-vision-ocr"


def image_paths(root):
    return sorted(p for p in Path(root).expanduser().iterdir() if p.suffix.lower() in EXTS)


def recognise_text(image_path):
    import Quartz
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(Path(image_path).resolve()))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(1)
    success, error = handler.performRequests_error_([request], None)
    if not success:
        raise RuntimeError(f"Vision request failed: {error}")
    lines = []
    for obs in request.results():
        candidates = obs.topCandidates_(1)
        if candidates:
            lines.append(candidates[0].string())
    return lines


def write_record(outdir, path, text, status, runtime):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{path.stem}.txt").write_text(text.rstrip() + ("\n" if text else ""))
    data = {
        "model_id": MODEL_ID,
        "runtime_seconds": round(runtime, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attempt_count": 1,
        "status": status,
    }
    (outdir / f"{path.stem}.json").write_text(json.dumps(data, indent=2) + "\n")


def run_image(path):
    start = time.perf_counter()
    lines = recognise_text(path)
    text = "\n".join(line.strip() for line in lines if line.strip())
    status = "ok" if text else "parse-failed"
    return text, status, time.perf_counter() - start


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="experiments")
    args = parser.parse_args(argv)
    outdir = Path(args.output).expanduser() / SLUG
    for path in image_paths(args.input):
        txt = outdir / f"{path.stem}.txt"
        js = outdir / f"{path.stem}.json"
        if txt.exists() and js.exists():
            print(path.stem, "exists", flush=True)
            continue
        text, status, runtime = run_image(path)
        write_record(outdir, path, text, status, runtime)
        print(path.stem, status, flush=True)


if __name__ == "__main__":
    main()
