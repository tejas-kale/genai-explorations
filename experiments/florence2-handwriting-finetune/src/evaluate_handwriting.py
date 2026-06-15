import argparse
import re
import string
from pathlib import Path

from jiwer import cer, wer


PUNCT = str.maketrans({c: " " for c in string.punctuation})


"""
Example, truth="hello world" and hypothesis="hello word":
- CER: one deleted character from 11, so 1 / 11 = 0.0909.
- WER: one wrong word from 2, so 1 / 2 = 0.5.
- Normalised WER: lower-case, remove punctuation, collapse whitespace, then WER.
"""


def text(path):
    return path.read_text().strip()


def normalise(value):
    return re.sub(r"\s+", " ", value.lower().translate(PUNCT)).strip()


def labels(path):
    return {p.stem: p for p in Path(path).expanduser().glob("*.txt")}


def model_rows(experiments, label_paths):
    for model_dir in sorted(Path(experiments).expanduser().iterdir()):
        if not model_dir.is_dir():
            continue
        rows = []
        for pred in sorted(model_dir.glob("*.txt")):
            label = label_paths.get(pred.stem)
            if label:
                truth = text(label)
                hypothesis = text(pred)
                rows.append({
                    "cer": cer(truth, hypothesis),
                    "wer": wer(truth, hypothesis),
                    "normalised_wer": wer(normalise(truth), normalise(hypothesis)),
                })
        if rows:
            yield model_dir.name, rows


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def summary(experiments, label_paths):
    out = []
    for model, rows in model_rows(experiments, label_paths):
        out.append({
            "model": model,
            "images": len(rows),
            "mean_cer": mean(rows, "cer"),
            "mean_wer": mean(rows, "wer"),
            "mean_normalised_wer": mean(rows, "normalised_wer"),
        })
    return sorted(out, key=lambda row: (row["mean_cer"], row["mean_wer"], row["model"]))


def print_summary(label_count, rows):
    print(f"labels: {label_count}")
    print()
    print(f"{'model':24} {'images':>6} {'mean_cer':>9} {'mean_wer':>9} {'mean_normalised_wer':>20}")
    for row in rows:
        print(f"{row['model']:24} {row['images']:6d} {row['mean_cer']:9.4f} {row['mean_wer']:9.4f} {row['mean_normalised_wer']:20.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--labels", default="labels")
    args = parser.parse_args(argv)
    label_paths = labels(args.labels)
    print_summary(len(label_paths), summary(args.experiments, label_paths))


if __name__ == "__main__":
    main()
