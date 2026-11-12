from __future__ import print_function, absolute_import
from PIL import Image
import numpy as np
import math
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    e.g. (32, 8, 3, 256, 128) = 32 tracklets, 8 frames each, 3 channels
    """
    sample_methods = ['dense', 'rrs_train', 'rrs_test']

    def __init__(self, dataset, seq_len=8, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len 
        self.sample = sample  
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        # Split the video into S segments; each segment is a sampling pool
        S = self.seq_len  # 8
        rgb_paths, event_paths, pid, camid, trackid = self.dataset[index]
        num = len(rgb_paths) # total number of frames

        # Restricted random sampling (RSS) preprocessing, shared by all strategies
        sample_clip = []
        frame_indices = list(range(num))
        if num < S:  # case 1: fewer frames than S
            strip = list(range(num)) + [frame_indices[-1]] * (S - num)
            for s in range(S):
                pool = strip[s * 1:(s + 1) * 1]
                sample_clip.append(list(pool))
        else:  # case 2: enough frames
            inter_val = math.ceil(num / S)
            strip = list(range(num)) + [frame_indices[-1]] * (inter_val * S - num)
            for s in range(S):
                pool = strip[inter_val * s:inter_val * (s + 1)]
                sample_clip.append(list(pool))
        sample_clip = np.array(sample_clip)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = list(range(num)) 
            indices_list = []

            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index += self.seq_len

            last_seq = frame_indices[cur_index:]

            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)

            indices_list.append(last_seq)  # <class 'list'>: [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 24, 25, 24, 25, 24, 25]]
            rgb_imgs_list = []
            event_imgs_list =[]

            for indices in indices_list:  # <class 'list'>: [0, 1, 2, 3, 4, 5, 6, 7]
                rgb_img_seq = []
                event_img_seq = []

                for index in indices:
                    index = int(index)
                    rgb_path = rgb_paths[index]
                    event_path = event_paths[index]
                    rgb_img = Image.open(rgb_path).convert('RGB')  
                    event_img = Image.open(event_path).convert('RGB')  
                    rgb_img_seq.append(rgb_img)
                    event_img_seq.append(event_img)

                seq = [rgb_img_seq, event_img_seq]

                if self.transform is not None:
                    seq = self.transform(seq)

                rgb_imgs = torch.stack(seq[0], dim=0) # (8, 3, 256, 128)
                event_imgs = torch.stack(seq[1], dim=0) 
                
                rgb_imgs_list.append(rgb_imgs)
                event_imgs_list.append(event_imgs)

            rgb_tensor = torch.stack(rgb_imgs_list) # (4, 8, 3, 256, 128)
            event_tensor = torch.stack(event_imgs_list)  # (4, 8, 3, 256, 128)

            return torch.stack([rgb_tensor, event_tensor]), pid, camid, trackid, ""

        elif self.sample == 'rrs_train':
            idx = np.random.choice(sample_clip.shape[1], sample_clip.shape[0])
            number = sample_clip[np.arange(len(sample_clip)), idx]

            rgb_paths = np.array(list(rgb_paths)) 
            event_paths = np.array(list(event_paths)) 

            rgb_img_seq = [Image.open(rgb_path).convert('RGB') for rgb_path in rgb_paths[number]]
            event_img_seq = [Image.open(event_path).convert('RGB') for event_path in event_paths[number]]

            seq = [rgb_img_seq, event_img_seq]
            if self.transform is not None:
                seq = self.transform(seq)

            rgb_tensor = torch.stack(seq[0], dim=0) # stack T RGB frames: (T, 3, 256, 128)
            event_tensor = torch.stack(seq[1], dim=0)  

            return torch.stack([rgb_tensor, event_tensor]), pid, camid, trackid, ""

        elif self.sample == 'rrs_test':
            number = sample_clip[:, 0]

            rgb_paths = np.array(list(rgb_paths))  
            event_paths = np.array(list(event_paths))  

            rgb_img_seq = [Image.open(rgb_path).convert('RGB') for rgb_path in rgb_paths[number]]
            event_img_seq = [Image.open(event_path).convert('RGB') for event_path in event_paths[number]]

            seq = [rgb_img_seq, event_img_seq]

            if self.transform is not None:
                seq = self.transform(seq)

            rgb_tensor = torch.stack(seq[0], dim=0) 
            event_tensor = torch.stack(seq[1], dim=0)  

            return torch.stack([rgb_tensor, event_tensor]), pid, camid, trackid, np.array([rgb_paths[number],event_paths[number]])
        
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


