#!/usr/bin/env bash
set -eux

curl -fsSL https://raw.githubusercontent.com/tejas-kale/genai-explorations/main/experiments/florence2-handwriting-finetune/src/hf/infer_florence.py -o /tmp/infer_florence.py
pip install -U 'transformers==4.41.2' datasets accelerate timm einops pillow huggingface_hub torch torchvision
python /tmp/infer_florence.py
