#!/usr/bin/env bash
set -eux

curl -fsSL https://raw.githubusercontent.com/tejas-kale/genai-explorations/main/experiments/florence2-handwriting-finetune/src/hf/train_florence.py -o /tmp/train_florence.py
pip install -U 'transformers<5' datasets accelerate timm einops pillow huggingface_hub torch torchvision
python /tmp/train_florence.py
