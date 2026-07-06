"""Sanitise a folder of handwriting scans before they leave this machine.

This is the first stage of the GCP transcription pipeline
(scripts/run_gcp_pipeline.sh calls this before creating any cloud
resources). It re-encodes every source image as a plain JPEG so that only
pixel data - never the original file - is uploaded to the remote VM:

  - Any EXIF metadata (GPS coordinates, device make/model, timestamps,
    thumbnails, etc.) embedded by a phone or scanner is dropped, because
    Image.save() to JPEG with a fresh PIL Image does not carry the source
    file's EXIF block along. That means no location/device metadata leaves
    the machine, even though the pipeline's network/IAM controls already
    keep the destination locked down.
  - HEIC/HEIF (the default format on iPhones) is converted to JPEG, so the
    remote VM only ever needs a JPEG decoder - one less codec dependency
    (pillow-heif) required in requirements-remote.txt on the VM side.
  - Images are downscaled to max_side on their longest edge, which keeps
    upload size and remote VRAM/processing time down without materially
    hurting OCR quality at typical handwriting-photo resolutions.

Output: sanitised JPEGs plus a manifest.json mapping each output file back
to its original source filename/stem, so downstream transcripts can be
matched back to the original scan.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def images(root):
    """Return all supported image files directly under root, sorted by name."""
    return sorted(p for p in Path(root).expanduser().iterdir() if p.suffix.lower() in EXTS)


def sanitise_image(src, dst, max_side=1600):
    """Decode src, strip its metadata, downscale it, and re-save as JPEG.

    - pillow_heif is registered (if installed) so HEIC/HEIF sources from
      iPhones can be opened at all; it's optional so this still works on
      environments that only ever see JPEG/PNG input.
    - ImageOps.exif_transpose() applies the EXIF orientation flag as an
      actual pixel rotation/flip before the EXIF data is discarded -
      otherwise a sanitised image with its orientation tag stripped could
      come out sideways.
    - convert("RGB") drops alpha/palette/CMYK modes so every output is a
      plain JPEG-compatible image.
    - Saving a fresh Image object as JPEG (quality=92) does not copy the
      source file's EXIF/metadata block, which is what actually strips the
      metadata - there is no explicit "remove EXIF" call because none is
      needed.
    """
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
