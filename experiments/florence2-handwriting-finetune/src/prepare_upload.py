import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def images(root):
    return sorted(p for p in Path(root).expanduser().iterdir() if p.suffix.lower() in EXTS)


def sanitise_image(src, dst, max_side=1600):
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    img.thumbnail((max_side, max_side))
    img.save(dst, format="JPEG", quality=92, optimize=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-side", default=1600, type=int)
    args = parser.parse_args(argv)
    out = Path(args.output).expanduser()
    manifest = []
    for src in images(args.input):
        dst = out / f"{src.stem}.jpg"
        sanitise_image(src, dst, args.max_side)
        manifest.append({"source": src.name, "file": dst.name, "stem": src.stem})
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
