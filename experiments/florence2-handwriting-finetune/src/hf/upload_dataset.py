import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Image


repo_id = "tejaskale/florence-handwriting-ocr"
root = Path(__file__).parents[2]
data = root / "data" / "train"

rows = [
    {"image": str(data / "images" / f"{p.stem}.jpg"), "text": p.read_text().strip(), "image_id": p.stem}
    for p in sorted((data / "labels").glob("*.txt"))
]

labelled = {p.stem for p in (root / "labels").glob("*.txt")}
image = next(p for p in sorted((root / "upload_sanitised").glob("*.jpg")) if p.stem not in labelled)
infer = Dataset.from_list([{"image": str(image), "text": "", "image_id": image.stem}]).cast_column("image", Image())
train = Dataset.from_list(rows).cast_column("image", Image())
DatasetDict({"train": train, "infer": infer}).push_to_hub(repo_id, private=True, token=os.environ["HF_TOKEN"])
