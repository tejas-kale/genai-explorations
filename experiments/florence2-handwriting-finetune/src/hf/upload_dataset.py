import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Image


repo_id = "tejaskale/florence-handwriting-ocr"
root = Path(__file__).parents[2]
data = root / "data" / "train"

rows = [
    {"image": str(data / "images" / f"{p.stem}.jpg"), "text": p.read_text().strip()}
    for p in sorted((data / "labels").glob("*.txt"))
]

ds = Dataset.from_list(rows).cast_column("image", Image())
DatasetDict({"train": ds}).push_to_hub(repo_id, private=True, token=os.environ["HF_TOKEN"])
