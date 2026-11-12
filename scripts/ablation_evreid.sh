#!/bin/bash
# Usage: bash scripts/ablation_evreid.sh [extra config overrides, e.g. MODEL.DEVICE_ID 1]
set -e
cd "$(dirname "$0")/.."
# Tables 4 and 6: ablations on EvReID (DINOv3). Full model = configs/EvReID/paths_dinov3.yml
for cfg in configs/EvReID/ablation/*.yml; do
  python train.py --config_file "$cfg" "$@"
done
