import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers.generation import GenerationMixin


token = os.environ["HF_TOKEN"]
dataset_id = os.getenv("DATASET_ID", "tejaskale/florence-handwriting-ocr")
model_id = os.getenv("MODEL_ID", "tejaskale/florence-handwriting-ocr-ft")
device = "cuda" if torch.cuda.is_available() else "cpu"
task = "<OCR>"

sample = load_dataset(dataset_id, token=token)["infer"][0]
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, token=token, attn_implementation="eager").to(device)
model.language_model.__class__ = type("Florence2LanguageModel", (model.language_model.__class__, GenerationMixin), {})
model.language_model.generation_config = GenerationConfig.from_model_config(model.language_model.config)

inputs = processor(text=task, images=sample["image"].convert("RGB"), return_tensors="pt").to(device)
ids = model.generate(**inputs, max_new_tokens=256)
predicted = processor.batch_decode(ids, skip_special_tokens=True)[0]
print({"image_id": sample["image_id"], "predicted": predicted})
