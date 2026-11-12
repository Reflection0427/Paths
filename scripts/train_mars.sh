#!/bin/bash
# Usage: bash scripts/train_mars.sh [extra config overrides, e.g. MODEL.DEVICE_ID 1]
set -e
cd "$(dirname "$0")/.."
# Table 2: MARS
python train.py --config_file configs/MARS/paths_dinov3.yml "$@"   # Paths† (DINOv3)
python train.py --config_file configs/MARS/paths_clip.yml "$@"     # Paths* (CLIP-B/16)
