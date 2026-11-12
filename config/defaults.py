from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'Paths'
# Pretrained weights for the DINOv3 / ViT backbones
_C.MODEL.PRETRAIN_PATH_T = 'pretrained/Dinov3.pth'
# OpenAI CLIP ViT-B/16 checkpoint for TRANSFORMER_TYPE 'ViT-B-16'
_C.MODEL.CLIP_PATH = 'pretrained/ViT-B-16.pt'
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
_C.MODEL.PROMPT = False  # optional CLIP visual prompts (off in all Paths configs)
_C.MODEL.ADAPTER = False  # optional CLIP adapters (off in all Paths configs)
_C.MODEL.FROZEN = False # whether to freeze the backbone

# ---- Paths modules (Sec. 3 of the paper) ----
_C.MODEL.USE_MAB = False  # Memory-Augmented Backbone (Sec. 3.1)
_C.MODEL.USE_PST = False  # Prompt-aware Spatio-temporal Transformer (Sec. 3.2)
_C.MODEL.USE_HMF = True   # Hierarchical Multi-modal Fusion: GGF + LTF (Sec. 3.3)
_C.MODEL.PST_SEQ_LEN = 4        # frames per tracklet seen by PST (T)
_C.MODEL.PST_START_LAYER = 0    # first transformer layer with temporal prompts
_C.MODEL.PST_PROMPT_LEN = 4     # number of temporal prompt tokens (G)
_C.MODEL.PST_PROMPT_GROUPS = 2  # number of prompt groups (g)
# ---- Loss weights ----
# Paper Eq.: L = L_ce + L_tri + lambda1 * L_pcl + lambda2 * L_align
_C.MODEL.MAB_LOSS_WEIGHT = 0.3        # lambda1, prototypical contrastive loss L_pcl (MAB)
_C.MODEL.MAB_MOMENTUM = 0.1  # momentum for updating the MAB identity prototypes (mu in the paper)
_C.MODEL.HMF_ALIGN_LOSS_WEIGHT = 0.1  # lambda2, cross-modal alignment loss L_align (HMF)
# Auxiliary terms used by the released implementation:
_C.MODEL.FRAME_LOSS_WEIGHT = 0.2      # frame-level ID loss on per-frame features
_C.MODEL.CM_SUPCON_LOSS_WEIGHT = 0.2  # cross-modal supervised contrastive loss


# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.HEAD = 12  # attention heads

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = True
_C.MODEL.SIE_VIEW = False  # We do not use this parameter


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
# Value of padding size
_C.INPUT.PADDING = 10
# Video ReID: Sequence length for video sequences
_C.INPUT.SEQ_LEN = 8
_C.INPUT.MIN_SEQ_LEN = 0
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./data')
_C.DATASETS.TRACK_INFO = ('./data')
_C.DATASETS.SPLIT_ID = 0
_C.DATASETS.EVENT_ROOT = ('./data')
# RGB degradation for PRID / iLIDS-VID: '' (clean), 'block', 'blur', 'dark'
_C.DATASETS.DEGRADE = ''



# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 14  # This may be affected by the order of data reading
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16  # You can adjust it to 8 to save memory while the batch_size need to be 64 to ensure the number of ID

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "SGD"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 120
# Base learning rate
_C.SOLVER.BASE_LR = 0.009
# Factor of learning bias
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 2
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
# Settings of range loss
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
# decay rate of learning rate
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 10
# method of warm up, option: 'constant','linear'

_C.SOLVER.SEED = 1111
_C.MODEL.NO_MARGIN = True
# epoch number of saving checkpoints
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 10
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 1
# Stop when mAP has not improved for this many evaluations (0 = off, train for MAX_EPOCHS)
_C.SOLVER.EARLY_STOP_PATIENCE = 0
# Evaluation strategy: evaluate first N epochs and last M epochs, skip middle epochs
# Set to 0 to disable (use EVAL_PERIOD instead)
_C.SOLVER.EVAL_FIRST_N_EPOCHS = 0  # Evaluate first N epochs (0 means disabled)
_C.SOLVER.EVAL_LAST_N_EPOCHS = 0   # Evaluate last N epochs (0 means disabled)
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 128  # You can adjust it to 64

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 256
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'before'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Use dense clip evaluation for final video-level testing.
# When enabled, val/test batch size is forced to 1 and clip features are averaged per tracklet.
_C.TEST.DENSE_EVAL = False
# ----------------------------------------------------------a------------------ #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./test"
# Record modifications made in this training run
_C.MODIFICATIONS = ""
