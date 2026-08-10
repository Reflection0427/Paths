# [ACMMM 2026] Paths: Prompt-aware Spatio-temporal Transformer with Hierarchical Multi-modal Fusion for RGB-Event Video Person Re-Identification

<p align="center">
  <a href="https://doi.org/10.1145/3767308.3835505"><img src="https://img.shields.io/badge/Paper-ACM%20MM%202026-blue" alt="Paper"></a>
  <a href="https://github.com/Reflection0427/Paths"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="Code"></a>
</p>

<p align="center">
  <a href="https://github.com/Reflection0427">Yakun Huo</a>,
  <a href="https://github.com/AWangYQ">Yingquan Wang</a>,
  <a href="https://github.com/010129">Yangyang Liu</a>,
  <a href="https://github.com/Drchip61">Tianyu Yan</a>,
  <a href="https://scholar.google.com/citations?user=-37EfvgAAAAJ">Yunzhi Zhuge</a>,
  <a href="https://scholar.google.com/citations?user=MfbIbuEAAAAJ">Pingping Zhang</a>,
  <a href="https://faculty.dlut.edu.cn/huchuan_Lu/en/index.htm">Huchuan Lu</a>
</p>

## Framework

<p align="center">
  <img src="assets/framework.png" width="100%" alt="Paths framework">
</p>



## Repository Structure

```text
Paths/
├── config/                 # Default configuration and compatibility aliases
├── configs/                # Dataset/backbone experiment configurations
│   ├── EvReID_Degrade/
│   ├── MARS/
│   ├── PRID/
│   └── iLIDSVID/
├── data/                   # Dataset parsers, video loader, transforms, sampler
├── engine/                 # Training and inference loops
├── layers/                 # ID, triplet, center, and auxiliary losses
├── modeling/
│   ├── backbones/          # ViT backbones
│   ├── clip/               # CLIP visual encoder
│   ├── dinov3/             # DINOv3 backbone code
│   ├── modules/
│   │   ├── mab/            # Memory-Augmented Backbone
│   │   ├── pst/            # Prompt-aware Spatio-temporal Transformer
│   │   └── hmf/            # Hierarchical Multi-modal Fusion
│   ├── backbone_router.py
│   └── paths_model.py      # Top-level Paths model
├── solver/                 # Optimizer and cosine scheduler
├── utils/                  # Metrics, logging, reranking, and utilities
├── visualize/              # Test-time visualization utilities
├── train.py
└── test.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Reflection0427/Paths.git
cd Paths
```

### 2. Create an environment

```bash
conda create -n paths python=3.10 -y
conda activate paths

# Install the PyTorch build that matches your CUDA driver first.
pip install torch torchvision

pip install yacs timm scipy scikit-learn pandas matplotlib seaborn \
    tqdm pillow ftfy regex omegaconf torchmetrics fvcore termcolor submitit
```

For a DINOv3 environment that enables optional optimized operators, install a compatible `xformers` build separately. It is not required by every Paths execution path.


## Dataset Preparation

### EvReID

```text
DATA_ROOT/
├── rgb_degrade/
│   ├── train/<pid>/*.jpg
│   ├── test/<pid>/*.jpg
│   └── info/
│       ├── train_name.txt
│       ├── test_name.txt
│       ├── tracks_train_info.mat
│       ├── tracks_test_info.mat
│       └── query_IDX.mat
└── event/
    ├── train/<pid>/*.jpg
    └── test/<pid>/*.jpg
```

Set `DATASETS.ROOT_DIR` to `DATA_ROOT/rgb_degrade`.

### MARS

```text
MARS_ROOT/
├── rgb/
│   ├── bbox_train/<pid>/*.jpg
│   └── bbox_test/<pid>/*.jpg
├── event/
│   ├── bbox_train/<pid>/*.jpg
│   └── bbox_test/<pid>/*.jpg
└── info/
    ├── train_name.txt
    ├── test_name.txt
    ├── tracks_train_info.mat
    ├── tracks_test_info.mat
    └── query_IDX.mat
```

Set `DATASETS.ROOT_DIR` to `MARS_ROOT`.

## Training

### EvReID with DINOv3

```bash
python train.py --config_file configs/EvReID_Degrade/Paths.yml \
    DATASETS.ROOT_DIR /absolute/path/to/EvReID/rgb_degrade \
    MODEL.PRETRAIN_PATH_T /absolute/path/to/Dinov3.pth \
    OUTPUT_DIR logs/EvReID_DINOv3
```

### EvReID with CLIP-B/16

```bash
python train.py --config_file configs/EvReID_Degrade/Paths.yml \
    MODEL.TRANSFORMER_TYPE ViT-B-16 \
    DATASETS.ROOT_DIR /absolute/path/to/EvReID/rgb_degrade \
    OUTPUT_DIR logs/EvReID_CLIP
```

### MARS with CLIP-B/16

```bash
python train.py --config_file configs/MARS/Paths.yml \
    DATASETS.ROOT_DIR /absolute/path/to/MARS \
    OUTPUT_DIR logs/MARS_CLIP
```

### MARS with DINOv3

```bash
python train.py --config_file configs/MARS/Path_Dino.yml \
    DATASETS.ROOT_DIR /absolute/path/to/MARS \
    MODEL.PRETRAIN_PATH_T /absolute/path/to/Dinov3.pth \
    OUTPUT_DIR logs/MARS_DINOv3
```

The best validation model is saved automatically as:

```text
<OUTPUT_DIR>/Paths_best.pth
```

## Evaluation

```bash
python test.py --config_file configs/EvReID_Degrade/Paths.yml \
    DATASETS.ROOT_DIR /absolute/path/to/EvReID/rgb_degrade \
    TEST.WEIGHT /absolute/path/to/Paths_best.pth \
    OUTPUT_DIR logs/EvReID_eval
```

## Citation

```bibtex
@inproceedings{huo2026paths,
  title     = {Paths: Prompt-aware Spatio-temporal Transformer with Hierarchical Multi-modal Fusion for RGB-Event Video Person Re-Identification},
  author    = {Huo, Yakun and Wang, Yingquan and Liu, Yangyang and Yan, Tianyu and Zhuge, Yunzhi and Zhang, Pingping and Lu, Huchuan},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3835505}
}
```
