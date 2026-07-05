#!/usr/bin/env bash
set -eux

curl -fsSL https://raw.githubusercontent.com/tejas-kale/genai-explorations/main/experiments/florence2-handwriting-finetune/src/hf/train_trocr_lines.py -o /tmp/train_trocr_lines.py
pip install -U 'transformers==5.10.2' sentencepiece datasets accelerate jiwer pillow huggingface_hub torch torchvision
python /tmp/train_trocr_lines.py
