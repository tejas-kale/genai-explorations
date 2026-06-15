import argparse
import csv
import json
from pathlib import Path

from jiwer import cer, wer


"""
Scores OCR transcripts against hand-corrected labels.

Example, using truth="hello world" and hypothesis="hello word":
- CER compares characters. "world" -> "word" deletes "l", so CER = 1 / 11 = 0.0909.
- WER compares words. One of two words is wrong, so WER = 1 / 2 = 0.5.
- Normalised WER first cleans text, e.g. "Hello,   world!" -> "hello world".
  If the hypothesis is "hello world", raw WER may penalise punctuation/case, while normalised WER is 0.

This script currently reports raw jiwer CER/WER. Add a normalisation transform if you want the
normalised variant in the output.
"""


def txt(path):
    # Strip file-edge whitespace so a trailing newline does not count as an OCR error.
    return path.read_text().strip()


def rows(experiments, labels):
    rows = []
    for model_dir in sorted(Path(experiments).expanduser().iterdir()):
        if not model_dir.is_dir():
            continue
        for pred in sorted(model_dir.glob("*.txt")):
            label = Path(labels).expanduser() / pred.name
            # Only score images with a hand-corrected label; unlabelled predictions are ignored.
            if label.exists():
                truth = txt(label)
                hypothesis = txt(pred)
                rows.append({
                    "model": model_dir.name,
                    "image_stem": pred.stem,
                    "cer": cer(truth, hypothesis),
                    "wer": wer(truth, hypothesis),
                    "label_chars": len(truth),
                    "prediction_chars": len(hypothesis),
                })
    return rows


def summary(rows):
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    out = []
    for model, items in by_model.items():
        # Mean per-image error rates make each image count equally in the model ranking.
        out.append({
            "model": model,
            "labelled_images": len(items),
            "mean_cer": sum(x["cer"] for x in items) / len(items),
            "mean_wer": sum(x["wer"] for x in items) / len(items),
        })
    return sorted(out, key=lambda x: (x["mean_cer"], x["mean_wer"], x["model"]))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep headers stable even when there are no scored files.
    keys = list(rows[0]) if rows else ["model", "image_stem", "cer", "wer", "label_chars", "prediction_chars"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(items):
    print("model labelled_images mean_cer mean_wer")
    for row in items:
        print(f"{row['model']} {row['labelled_images']} {row['mean_cer']:.4f} {row['mean_wer']:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--labels", default="labels")
    parser.add_argument("--output", default="eval_out")
    args = parser.parse_args(argv)
    out = Path(args.output).expanduser()
    breakdown = rows(args.experiments, args.labels)
    ranked = summary(breakdown)
    out.mkdir(parents=True, exist_ok=True)
    (out / "breakdown.json").write_text(json.dumps(breakdown, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(ranked, indent=2) + "\n")
    write_csv(out / "breakdown.csv", breakdown)
    write_csv(out / "summary.csv", ranked)
    print_summary(ranked)


if __name__ == "__main__":
    main()
