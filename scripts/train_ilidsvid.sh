#!/bin/bash
# Usage: bash scripts/train_ilidsvid.sh [extra config overrides, e.g. MODEL.DEVICE_ID 1]
set -e
cd "$(dirname "$0")/.."
# Table 3: iLIDS-VID (Blur / Occlusion)
for backbone in dinov3 clip; do
  for setting in blur occlusion; do
    python train.py --config_file configs/iLIDSVID/paths_${backbone}_${setting}.yml "$@"
  done
done
