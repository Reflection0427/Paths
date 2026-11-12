import torch
import random
import math
import numpy as np
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Compose(object):
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seqs):
        for t in self.transforms:
            seqs = t(seqs)
        return seqs


class Resize(object):
    """
    Sequence version of T.Resize(cfg.INPUT.SIZE_TRAIN).
    Resizes every frame to (h, w).
    """
    def __init__(self, h, w, interpolation=Image.BILINEAR):
        self.height = h
        self.width = w
        self.interpolation = interpolation
    def __call__(self, seqs):
            # seqs structure: [[RGB_frames], [Event_frames]]
            new_seqs = []
            for modal in seqs:
                new_modal = []
                for frame in modal:
                    # resize only when the size differs
                    if frame.size[0] != self.width or frame.size[1] != self.height:
                        new_modal.append(frame.resize((self.width, self.height), self.interpolation))
                    else:
                        new_modal.append(frame)
                new_seqs.append(new_modal)
            return new_seqs


class Pad(object):
    """
    Sequence version of T.Pad(cfg.INPUT.PADDING).
    Pads every frame on all sides.
    """
    def __init__(self, padding=10, fill=0, padding_mode='constant'):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, seqs):
        new_seqs = []
        for modal in seqs:
            new_modal = []
            for frame in modal:
                # pad with ImageOps.expand
                # padding is an int (all sides) or a tuple (left, top, right, bottom)
                new_frame = ImageOps.expand(frame, border=self.padding, fill=self.fill)
                new_modal.append(new_frame)
            new_seqs.append(new_modal)
        return new_seqs


class RandomCrop(object):
    """
    Sequence version of T.RandomCrop(cfg.INPUT.SIZE_TRAIN).
    The crop box is sampled once and applied to all frames and modalities, keeping them aligned.
    """
    def __init__(self, h, w):
        self.height = h
        self.width = w

    def __call__(self, seqs):
        # 1. image size (all frames share it, so use the first frame)
        # note: frames are already padded, so they are larger than (h, w)
        img_w, img_h = seqs[0][0].size
        
        # 2. sample the crop position once
        if img_w == self.width and img_h == self.height:
            x1, y1 = 0, 0
        else:
            # keep the crop box inside the image
            x1 = random.randint(0, max(0, img_w - self.width))
            y1 = random.randint(0, max(0, img_h - self.height))

        # 3. apply to every frame of every modality
        new_seqs = []
        for modal in seqs:
            new_modal = []
            for frame in modal:
                frame = frame.crop((x1, y1, x1 + self.width, y1 + self.height))
                new_modal.append(frame)
            new_seqs.append(new_modal)
        
        return new_seqs

class RandomHorizontalFlip(object):
    """
    Sequence version of T.RandomHorizontalFlip(p=cfg.INPUT.PROB).
    The flip decision is sampled once and applied to all frames.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, seqs):
        # sample once
        if random.random() < self.p:
            new_seqs = []
            for modal in seqs:
                new_modal = []
                for frame in modal:
                    new_modal.append(frame.transpose(Image.FLIP_LEFT_RIGHT))
                new_seqs.append(new_modal)
            return new_seqs
        return seqs

class ToTensor(object):
    """
    Converts lists of PIL images to lists of tensors.
    """
    def __call__(self, seqs):
        new_seqs = []
        for modal in seqs:
            new_modal = []
            for frame in modal:
                # handle different image modes
                pic = frame
                if pic.mode == 'I':
                    img = torch.from_numpy(np.array(pic, np.int32, copy=False))
                elif pic.mode == 'I;16':
                    img = torch.from_numpy(np.array(pic, np.int16, copy=False))
                elif pic.mode == 'YCbCr':
                    nchannel = 3
                    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
                    img = img.view(pic.size[1], pic.size[0], nchannel)
                else:
                    # default: RGB
                    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
                    # view: (H, W, C)
                    img = img.view(pic.size[1], pic.size[0], len(pic.mode))
                
                # put it from HWC to CHW format
                img = img.transpose(0, 1).transpose(0, 2).contiguous()
                if isinstance(img, torch.ByteTensor):
                    img = img.float().div(255)
                
                new_modal.append(img)
            new_seqs.append(new_modal)
        return new_seqs

class Normalize(object):
    """
    Sequence version of T.Normalize.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, seqs):
        # seqs are lists of tensors here
        for modal in seqs:
            for frame in modal: # frame is Tensor [C, H, W]
                for t, m, s in zip(frame, self.mean, self.std):
                    t.sub_(m).div_(s)
        return seqs


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class VideoRandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5,
            min_area=0.02,
            max_area=1 / 3,
            min_aspect=0.3,
            max_aspect=None,
            mode='const',
            min_count=1,
            max_count=None,
            num_splits=0,
            device='cuda',
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == 'rand':
            self.rand_color = True  # per block random normal
        elif self.mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not self.mode or self.mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def __call__(self, seqs):
        # if len(input.size()) == 3:
        #     self._erase(input, *input.size(), input.dtype)
        # else:
        #     batch_size, chan, img_h, img_w = input.size()
        #     # skip first slice of batch if num_splits is set (for clean portion of samples)
        #     batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
        #     for i in range(batch_start, batch_size):
        #         self._erase(input[i], chan, img_h, img_w, input.dtype)
        # return input
        """
        Works on the nested list structure [[Tensor...], [Tensor...]].
        """
        # for each modality (RGB list, Event list)
        for modal_list in seqs:
            # for each frame (Tensor [C, H, W])
            for img in modal_list:
                # apply the original erasing to a single tensor
                self._erase(img, *img.size(), img.dtype)
        
        return seqs

    def __repr__(self):
        # NOTE simplified state for repr
        fs = self.__class__.__name__ + f'(p={self.probability}, mode={self.mode}'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs