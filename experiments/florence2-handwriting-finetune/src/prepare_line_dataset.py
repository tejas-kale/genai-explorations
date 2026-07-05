"""Build a line-level TrOCR dataset from the labelled full-page transcripts.

Full-page fine-tuning (train_trocr.py) memorises page *layout* rather than
learning to read handwriting: with ~20 unique full-page images the model can
trivially associate an image with "the transcript for that specific photo"
without ever generalising to new handwriting. Splitting every page into
individual handwritten *lines* turns ~20 pages into several hundred short,
mostly-distinct (image, text) pairs, which is a much harder-to-memorise,
more IAM-like training signal.

IMPORTANT DEVIATION FROM THE ORIGINAL SPEC (read this before touching the
matching logic below): the original plan was to segment each page into
physical ink-row crops and pair them 1:1 with the label file's non-blank
lines whenever the two counts matched exactly (skipping the page after two
relaxed segmentation retries otherwise). That assumes a label "line" is one
physical handwritten row. Direct inspection of the source photos (e.g.
upload_sanitised/IMG_5472.jpg, IMG_5498.jpg) shows this is false: a label
line is a *logical* sentence/bullet that commonly wraps across 2-4 physical
ink rows, and re-running the default (tuned) segmenter against every one of
the 30 labelled pages found it NEVER produces a physical-row count equal to
the label-line count -- only forced/relaxed variants occasionally hit the
same integer by coincidence. Visually inspecting one such coincidental match
(IMG_5498) confirmed the crops were misaligned to the wrong label lines from
index 1 onward despite the totals matching, i.e. the 1:1-count-match
approach silently produces mislabelled pairs.

Because of this, the pairing algorithm here instead:
  1. Segments the page into physical ink-row crops (reusing
     scripts/trocr_eval.py's tuned page-crop + adaptive-binarisation +
     projection-profile segmentation, retried with up to two relaxed
     variants if the default under-segments below the label line count).
  2. Allocates those physical rows across the label lines proportionally to
     each label line's character length (largest-remainder rounding, at
     least one row per label line), and merges each label line's assigned
     rows into a single composite crop (vertical stack, in reading order).
  3. Skips the page only if even the most relaxed variant still finds fewer
     physical rows than label lines (there is nothing to allocate).
This directly encodes what the photos show (short label lines usually get
one row; long/wrapped ones get several) instead of forcing false 1:1 crop
counts. It is an approximation -- row boundaries are allocated proportional
to text length, not detected precisely -- but it was visually verified (see
save_sanity_sample) to correctly align crops to their paired text, which the
original count-matching approach did not.

Usage:
  python src/prepare_line_dataset.py                 # build + report + sanity sample only
  python src/prepare_line_dataset.py --push           # also push to the HF Hub
"""
import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import trocr_eval  # noqa: E402  (needs ROOT on sys.path first)

# Fixed page-level split, taken from the completed full-page training run.
# Do NOT re-derive this: it must stay identical to the split train_trocr.py
# already reported results against, so line-level results are comparable.
TRAIN_PAGES = [
    "IMG_5491", "IMG_5486", "IMG_5482", "IMG_5498", "IMG_5494", "IMG_5478",
    "IMG_5477", "IMG_5484", "IMG_5483", "IMG_5487", "IMG_5481", "IMG_5497",
    "IMG_5501", "IMG_5493", "IMG_5488", "IMG_5499", "IMG_5473", "IMG_5485",
    "IMG_5490", "IMG_5474", "IMG_5489",
]
HOLDOUT_PAGES = [
    "IMG_5500", "IMG_5476", "IMG_5496", "IMG_5479", "IMG_5480", "IMG_5495",
    "IMG_5472", "IMG_5475", "IMG_5492",
]

DATA_DIR = ROOT / "data"
LABELS_DIR = ROOT / "labels"
UPLOAD_SANITISED_DIR = ROOT / "upload_sanitised"

REPO_ID = "tejaskale/trocr-handwriting-lines"

# Segmentation variants to retry, in order, when the default tuned constants
# find FEWER physical ink rows than the page has label lines (each label
# line needs at least one row to allocate). Only the line-*segmentation*
# knobs are overridden -- the ink mask itself (computed once per page) is
# reused across variants. There is no "merge more aggressively" variant
# here (unlike an earlier version of this script): over-segmentation isn't
# a failure mode any more, since extra rows just get allocated to whichever
# label line they proportionally belong to.
SEGMENTATION_VARIANTS = [
    {},  # default: scripts/trocr_eval.py's tuned constants, unmodified
    {   # split more finely: fixes under-segmentation (too few rows)
        "MIN_GAP_PX": 3,
        "MAX_LINE_HEIGHT_PX": 70,
        "LINE_ROW_FRAC": 0.0015,
    },
    {   # split even more finely
        "MIN_GAP_PX": 2,
        "MAX_LINE_HEIGHT_PX": 50,
        "LINE_ROW_FRAC": 0.001,
        "MIN_LINE_HEIGHT_PX": 5,
    },
]


@contextmanager
def overridden_constants(overrides):
    """Temporarily monkeypatch module-level constants on trocr_eval.

    trocr_eval.segment_lines() reads its tuning knobs (MIN_GAP_PX etc.) as
    bare module globals, resolved at call time -- so patching them on the
    module object changes segment_lines()'s behaviour for calls made while
    the patch is active, without needing to fork or parametrise that code.
    """
    originals = {name: getattr(trocr_eval, name) for name in overrides}
    for name, value in overrides.items():
        setattr(trocr_eval, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(trocr_eval, name, value)


def load_grayscale_any(image_path):
    """Load a page image (JPEG or HEIC) as a TARGET_WIDTH-downscaled grayscale
    array, matching trocr_eval.load_grayscale's HEIC path so the same
    pixel-tuned segmentation constants apply regardless of source format.
    """
    if image_path.suffix.lower() in (".jpg", ".jpeg"):
        img = PILImage.open(image_path).convert("L")
        scale = trocr_eval.TARGET_WIDTH / img.width
        img = img.resize(
            (trocr_eval.TARGET_WIDTH, round(img.height * scale)), PILImage.LANCZOS
        )
        import numpy as np
        return np.asarray(img).astype(np.float64)
    return trocr_eval.load_grayscale(image_path)


def resolve_image_path(stem):
    jpg_path = UPLOAD_SANITISED_DIR / f"{stem}.jpg"
    if jpg_path.exists():
        return jpg_path
    heic_path = DATA_DIR / f"{stem}.HEIC"
    if heic_path.exists():
        return heic_path
    return None


def read_label_lines(label_path):
    """Non-empty physical lines, trailing whitespace stripped. Blank lines
    are paragraph separators in the transcript convention, not ink lines."""
    text = label_path.read_text()
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def segment_page(image_path):
    """Return (page_gray, ink_mask) shared across all segmentation variants
    for this page, so re-segmenting doesn't redo the page-crop/ink-mask work."""
    gray = load_grayscale_any(image_path)
    page = trocr_eval.crop_to_page(gray)
    ink = trocr_eval.ink_mask(page)
    return page, ink


def get_sufficient_row_crops(page, ink, min_rows):
    """Try the default segmentation, then relaxed (finer-splitting) variants,
    until the physical-row crop count is >= min_rows (every label line needs
    at least one row to allocate). Returns (crops, attempts); crops is None
    if even the most relaxed variant still finds too few rows.
    """
    attempts = []
    for variant in SEGMENTATION_VARIANTS:
        with overridden_constants(variant):
            crops = trocr_eval.segment_lines(page, ink)
        attempts.append(len(crops))
        if len(crops) >= min_rows:
            return crops, attempts
    return None, attempts


def allocate_rows(lengths, total_rows):
    """Split `total_rows` physical ink rows across len(lengths) label lines,
    proportional to each label line's character length, at least one row
    each, using largest-remainder rounding so the allocation sums exactly
    to total_rows. Encodes the observed reality that longer/wrapped label
    lines span more physical rows than short ones.
    """
    n = len(lengths)
    total_chars = sum(lengths)
    raw = [total_rows / n] * n if total_chars == 0 else [total_rows * l / total_chars for l in lengths]
    alloc = [max(1, int(r)) for r in raw]
    # Bumping every floor(raw[i]) < 1 up to the 1-row minimum can push the
    # total above total_rows (e.g. many short label lines on the same page
    # as a few long ones); largest-remainder-first covers the shortfall
    # case (diff > 0), smallest-remainder-first (among entries with room to
    # shrink) covers the overshoot case (diff < 0).
    diff = total_rows - sum(alloc)
    if diff > 0:
        order = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for i in range(diff):
            alloc[order[i % n]] += 1
    elif diff < 0:
        order = sorted(range(n), key=lambda i: raw[i] - int(raw[i]))
        i = 0
        remaining = -diff
        guard = 0
        while remaining > 0 and guard < 10 * n + 10:
            idx = order[i % n]
            if alloc[idx] > 1:
                alloc[idx] -= 1
                remaining -= 1
            i += 1
            guard += 1
    return alloc


def build_composite_crops(row_crops, alloc):
    """Group consecutive row crops per `alloc` counts and vertically stack
    each group (in reading order) into one composite crop per label line."""
    import numpy as np

    groups = []
    idx = 0
    for count in alloc:
        group = row_crops[idx: idx + count]
        idx += count
        groups.append(group[0] if len(group) == 1 else np.vstack(group))
    return groups


def match_page(stem, image_path, label_lines):
    """Segment the page into physical ink rows, then allocate rows to label
    lines proportionally (see module docstring). Returns (crops, attempts,
    alloc); crops is None if even the most relaxed variant found fewer rows
    than label lines.
    """
    page, ink = segment_page(image_path)
    row_crops, attempts = get_sufficient_row_crops(page, ink, len(label_lines))
    if row_crops is None:
        return None, attempts, None
    alloc = allocate_rows([len(l) for l in label_lines], len(row_crops))
    crops = build_composite_crops(row_crops, alloc)
    return crops, attempts, alloc


def build_line_image(line_arr):
    """Grayscale line crop -> contrast-normalised RGB image with a white
    border, exactly as trocr_eval.prepare_for_ocr feeds TrOCR at inference,
    so training sees the same distribution eval/inference will use."""
    return trocr_eval.prepare_for_ocr(line_arr)


def process_pages(page_ids):
    """Returns (rows, matched, skipped) for the given page ids.

    rows: list of dicts {page_id, line_index, text, image (PIL.Image)}
    matched: list of page ids that paired 1:1
    skipped: list of {page_id, reason, label_line_count, attempts}
    """
    rows = []
    matched = []
    skipped = []
    for stem in page_ids:
        label_path = LABELS_DIR / f"{stem}.txt"
        image_path = resolve_image_path(stem)
        if image_path is None or not label_path.exists():
            skipped.append({
                "page_id": stem,
                "reason": "missing_image_or_label",
                "label_line_count": None,
                "attempts": [],
            })
            continue

        label_lines = read_label_lines(label_path)
        crops, attempts, alloc = match_page(stem, image_path, label_lines)
        if crops is None:
            skipped.append({
                "page_id": stem,
                "reason": "insufficient_physical_rows",
                "label_line_count": len(label_lines),
                "attempts": attempts,
            })
            continue

        matched.append(stem)
        for line_index, (crop, text) in enumerate(zip(crops, label_lines)):
            rows.append({
                "page_id": stem,
                "line_index": line_index,
                "text": text,
                "image": build_line_image(crop),
            })
    return rows, matched, skipped


def save_sanity_sample(rows, matched_pages, out_dir, num_pages=2, num_lines=3):
    """Write the first `num_lines` line crops (+ their paired text) for the
    first `num_pages` matched pages to `out_dir`, for visual inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_pages = matched_pages[:num_pages]
    texts_out = []
    for stem in sample_pages:
        page_rows = [r for r in rows if r["page_id"] == stem][:num_lines]
        for r in page_rows:
            png_path = out_dir / f"{stem}_line{r['line_index']}.png"
            r["image"].save(png_path)
            texts_out.append(f"{png_path.name}: {r['text']!r}")
    (out_dir / "texts.txt").write_text("\n".join(texts_out) + "\n")
    return sample_pages, texts_out


def build_report(train_rows, train_matched, train_skipped, holdout_rows, holdout_matched, holdout_skipped):
    return {
        "train": {
            "pages_matched": train_matched,
            "pages_skipped": train_skipped,
            "num_pages_matched": len(train_matched),
            "num_pages_skipped": len(train_skipped),
            "num_line_pairs": len(train_rows),
        },
        "holdout": {
            "pages_matched": holdout_matched,
            "pages_skipped": holdout_skipped,
            "num_pages_matched": len(holdout_matched),
            "num_pages_skipped": len(holdout_skipped),
            "num_line_pairs": len(holdout_rows),
        },
    }


def push_dataset(train_rows, holdout_rows, repo_id, token):
    from datasets import Dataset, DatasetDict, Features, Image, Value

    features = Features({
        "image": Image(),
        "text": Value("string"),
        "page_id": Value("string"),
        "line_index": Value("int32"),
    })

    def to_dataset(rows):
        cols = {
            "image": [r["image"] for r in rows],
            "text": [r["text"] for r in rows],
            "page_id": [r["page_id"] for r in rows],
            "line_index": [r["line_index"] for r in rows],
        }
        return Dataset.from_dict(cols, features=features)

    dataset = DatasetDict({
        "train": to_dataset(train_rows),
        "holdout": to_dataset(holdout_rows),
    })
    dataset.push_to_hub(repo_id, private=True, token=token)
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Push the built dataset to the HF Hub.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument(
        "--sanity-dir",
        default="/private/tmp/claude-501/-Users-tejaskale-Code-genai-explorations-experiments-florence2-handwriting-finetune/52eb7d87-2839-44f8-a372-615066714caa/scratchpad/line_sanity",
    )
    parser.add_argument(
        "--report-path",
        default="/private/tmp/claude-501/-Users-tejaskale-Code-genai-explorations-experiments-florence2-handwriting-finetune/52eb7d87-2839-44f8-a372-615066714caa/scratchpad/line_dataset_report.json",
    )
    args = parser.parse_args(argv)

    print("processing train pages...")
    train_rows, train_matched, train_skipped = process_pages(TRAIN_PAGES)
    print("processing holdout pages...")
    holdout_rows, holdout_matched, holdout_skipped = process_pages(HOLDOUT_PAGES)

    report = build_report(
        train_rows, train_matched, train_skipped,
        holdout_rows, holdout_matched, holdout_skipped,
    )
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved report to {report_path}")

    sanity_dir = Path(args.sanity_dir)
    sample_pages, texts_out = save_sanity_sample(train_rows, train_matched, sanity_dir)
    print(f"sanity sample pages: {sample_pages}")
    for line in texts_out:
        print(f"  {line}")
    print(f"sanity crops saved to {sanity_dir}")

    if args.push:
        import os
        token = os.environ["HF_TOKEN"]
        push_dataset(train_rows, holdout_rows, args.repo_id, token)
        print(f"pushed dataset to {args.repo_id}: train={len(train_rows)} holdout={len(holdout_rows)}")


if __name__ == "__main__":
    main()
