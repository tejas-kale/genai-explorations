import argparse
import html
import io
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


@dataclass
class State:
    images: Path
    experiments: Path
    labels: Path
    draft_model: str

    def __post_init__(self):
        self.images = Path(self.images).expanduser()
        self.experiments = Path(self.experiments).expanduser()
        self.labels = Path(self.labels).expanduser()

    def items(self):
        return sorted(p for p in self.images.iterdir() if p.suffix.lower() in EXTS)

    def image(self, stem):
        for path in self.items():
            if path.stem == stem:
                return path
        raise FileNotFoundError(stem)

    def text_for(self, stem):
        label = self.labels / f"{stem}.txt"
        draft = self.experiments / self.draft_model / f"{stem}.txt"
        if label.exists():
            return label.read_text()
        if draft.exists():
            return draft.read_text()
        return ""

    def save(self, stem, text):
        self.labels.mkdir(parents=True, exist_ok=True)
        (self.labels / f"{stem}.txt").write_text(text)


def open_image(path):
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    try:
        return Image.open(path)
    except UnidentifiedImageError:
        if path.suffix.lower() not in {".heic", ".heif"}:
            raise
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", f.name], check=True)
            return Image.open(f.name).copy()


def jpeg_bytes(path):
    img = ImageOps.exif_transpose(open_image(path)).convert("RGB")
    img.thumbnail((1600, 1600))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


def page(state, index):
    items = state.items()
    index = min(max(index, 0), max(len(items) - 1, 0))
    if not items:
        return "<h1>No images found</h1>"
    item = items[index]
    prev_i = max(index - 1, 0)
    next_i = min(index + 1, len(items) - 1)
    text = html.escape(state.text_for(item.stem))
    stem = html.escape(item.stem)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Handwriting labels</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; }}
header {{ padding: 12px 18px; background: #111; color: white; display: flex; gap: 16px; align-items: center; }}
main {{ display: grid; grid-template-columns: 1fr 1fr; height: calc(100vh - 52px); }}
figure {{ margin: 0; padding: 18px; overflow: auto; background: #f4f4f4; }}
img {{ max-width: 100%; height: auto; box-shadow: 0 2px 12px #999; }}
form {{ padding: 18px; display: flex; flex-direction: column; gap: 12px; }}
textarea {{ flex: 1; font: 18px ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1.45; padding: 12px; }}
button, a {{ font-size: 16px; }}
nav {{ margin-left: auto; display: flex; gap: 12px; }}
</style></head>
<body><header><strong>{stem}</strong><span>{index + 1} / {len(items)}</span><nav><a href="/?i={prev_i}">Prev</a><a href="/?i={next_i}">Next</a></nav></header>
<main><figure><img src="/image/{urllib.parse.quote(item.stem)}"></figure>
<form method="post" action="/save"><input type="hidden" name="stem" value="{stem}"><input type="hidden" name="i" value="{index}"><textarea name="text" autofocus>{text}</textarea><button>Save label</button></form></main>
</body></html>"""


def handler(state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path.startswith("/image/"):
                stem = urllib.parse.unquote(parsed.path.removeprefix("/image/"))
                data = jpeg_bytes(state.image(stem))
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            params = urllib.parse.parse_qs(parsed.query)
            body = page(state, int(params.get("i", [0])[0])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            data = urllib.parse.parse_qs(self.rfile.read(size).decode())
            state.save(data["stem"][0], data.get("text", [""])[0])
            i = data.get("i", ["0"])[0]
            self.send_response(303)
            self.send_header("Location", f"/?i={i}")
            self.end_headers()

        def log_message(self, format, *args):
            return
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data")
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--labels", default="labels")
    parser.add_argument("--draft-model", default="qwen3.5-9b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)
    state = State(Path(args.images), Path(args.experiments), Path(args.labels), args.draft_model)
    server = ThreadingHTTPServer((args.host, args.port), handler(state))
    print(f"http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
