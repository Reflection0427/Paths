import torch
from torch.utils.data import DataLoader
import utils.seqtransforms as SeqT
from .video_loader_xh import VideoDataset
from .samplers import RandomIdentitySampler
from .EvReID import EvReID
from .MARS import MARS
from .iLIDSVID import iLIDSVID
__factory = {
	'EvReID': EvReID,
    'MARS': MARS,
    'iLIDSVID': iLIDSVID
}



def build_dataset(cfg):
	name = cfg.DATASETS.NAMES
	if name == 'iLIDSVID':
		return iLIDSVID(root=cfg.DATASETS.ROOT_DIR, event_root=cfg.DATASETS.EVENT_ROOT,
						degrade=cfg.DATASETS.DEGRADE, split_id=cfg.DATASETS.SPLIT_ID)
	return __factory[name](root=cfg.DATASETS.ROOT_DIR)

def train_collate_fn(batch):
	"""
    # The input of collate_fn is a list of length batch_size; each element is one __getitem__ result
    """
	imgs, pids, camids, viewids, _ = zip(*batch)
	pids = torch.tensor(pids, dtype=torch.int64)
	viewids = torch.tensor(viewids, dtype=torch.int64)
	camids = torch.tensor(camids, dtype=torch.int64)
	# Stack into tensors, then index RGB and EVENT
	imgs = torch.stack(imgs, dim=0)  # (batch_size, 2, seq_len, 3, H, W)
	RGB = imgs[:, 0]  # (batch_size, seq_len, 3, H, W)
	EVENT = imgs[:, 1]  # (batch_size, seq_len, 3, H, W)	
	imgs = {'RGB': RGB, "EVENT": EVENT}
	return imgs, pids, camids, viewids, _


def val_collate_fn(batch):
	imgs, pids, camids, viewids, img_paths = zip(*batch)
	viewids = torch.tensor(viewids, dtype=torch.int64)
	camids_batch = torch.tensor(camids, dtype=torch.int64)
	# Stack into tensors, then index RGB and EVENT
	imgs = torch.stack(imgs, dim=0)  # (batch_size, 2, seq_len, 3, H, W)
	RGB = imgs[:, 0]  # (batch_size, seq_len, 3, H, W)
	EVENT = imgs[:, 1]  # (batch_size, seq_len, 3, H, W)
	imgs = {'RGB': RGB, "EVENT": EVENT}
	return imgs, pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):		
	seq_len = cfg.INPUT.SEQ_LEN
	num_workers = cfg.DATALOADER.NUM_WORKERS

	dataset = build_dataset(cfg)

	train_transforms = SeqT.Compose([
		SeqT.Resize(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
		SeqT.RandomHorizontalFlip(p=cfg.INPUT.PROB),
		SeqT.Pad(padding=cfg.INPUT.PADDING),
		SeqT.RandomCrop(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
		SeqT.ToTensor(),
		SeqT.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
		SeqT.VideoRandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')
	])

	val_transforms = SeqT.Compose([
		SeqT.Resize(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
	    SeqT.ToTensor(),
	    SeqT.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
	])

	train_set = VideoDataset(dataset.train, seq_len=seq_len, sample='rrs_train', transform=train_transforms)
	train_set_normal = VideoDataset(dataset.train, seq_len=seq_len, sample='dense', transform=val_transforms)
	use_dense_eval = bool(getattr(cfg.TEST, 'DENSE_EVAL', False))
	val_sample = 'dense' if use_dense_eval else 'rrs_test'
	val_set = VideoDataset(dataset.query + dataset.gallery, seq_len=seq_len, sample=val_sample, transform=val_transforms)

	num_classes = dataset.num_train_pids  # 625
	cam_num = dataset.num_train_cams  # 6
	view_num = 1  # 1

	# DataLoader optimization parameters
	loader_kwargs = {
		'pin_memory': True,
		'persistent_workers': True,
		'prefetch_factor': 2,
	}

	train_loader = DataLoader(
		train_set,
		sampler=RandomIdentitySampler(dataset.train, batch_size=cfg.SOLVER.IMS_PER_BATCH, num_instances=cfg.DATALOADER.NUM_INSTANCE),
		batch_size=cfg.SOLVER.IMS_PER_BATCH,
		num_workers=num_workers,
		drop_last=True,
		collate_fn=train_collate_fn,
		**loader_kwargs
	)

	train_loader_normal = DataLoader(
		train_set_normal,
		batch_size=1,
		shuffle=True,
		num_workers=num_workers,
		drop_last=True,
		collate_fn=train_collate_fn,
		**loader_kwargs
	)

	val_loader = DataLoader(
		val_set,
		batch_size=1 if use_dense_eval else cfg.TEST.IMS_PER_BATCH,
		shuffle=False,
		num_workers=num_workers,
		drop_last=False,
		collate_fn=val_collate_fn,
		**loader_kwargs
	)

	return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num


def make_eval_all_dataloader(cfg):
	seq_len = cfg.INPUT.SEQ_LEN
	num_workers = cfg.DATALOADER.NUM_WORKERS

	dataset = build_dataset(cfg)

	transform_test = SeqT.Compose([
		SeqT.Resize(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
		SeqT.ToTensor(),
		SeqT.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
	])

	num_classes = dataset.num_train_pids
	cam_num = dataset.num_train_cams
	view_num = dataset.num_train_vids

	sampler_method = 'dense'
	batch_size_eval = 1

	val_set = VideoDataset(dataset.query + dataset.gallery, seq_len=seq_len, sample=sampler_method,
	                       transform=transform_test)

	# DataLoader optimization parameters
	loader_kwargs = {
		'pin_memory': True,
		'persistent_workers': True,
		'prefetch_factor': 2,
	}

	val_loader = DataLoader(
		val_set,
		batch_size=batch_size_eval,
		shuffle=False,
		num_workers=num_workers,
		drop_last=False,
		collate_fn=val_collate_fn,
		**loader_kwargs
	)

	return val_loader, len(dataset.query), num_classes, cam_num, view_num
