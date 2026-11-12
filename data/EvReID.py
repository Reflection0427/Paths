from __future__ import print_function, absolute_import
import os.path as osp
from scipy.io import loadmat
import numpy as np
from utils.serialization import write_json, read_json

class infostruct(object):
    pass

class EvReID(object):
    def __init__(self, root='', min_seq_len=0):
        self.root = root
        self.train_name_path = osp.join(self.root, 'info/train_name.txt')
        self.test_name_path = osp.join(self.root, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self.root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self.root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self.root, 'info/query_IDX.mat')
        self.split_train_json_path = osp.join(self.root, 'split_train_clip.json')
        self.split_query_json_path = osp.join(self.root, 'split_query_clip.json')
        self.split_gallery_json_path = osp.join(self.root, 'split_gallery_clip.json')
        # Check that files exist
        self._check_before_run()

        # Load metadata
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)

        track_train_mat = loadmat(self.track_train_info_path)
        # track_train: numpy.ndarray (8298, 4)
        # columns: [start_frame, end_frame, person_id, cam_id]
        track_train = np.array([track_train_mat['start_frame'], track_train_mat['end_frame'], 
                                track_train_mat['person_id'], track_train_mat['cam_id']]).squeeze(1).transpose(1,0)
        track_test_mat = loadmat(self.track_test_info_path) # numpy.ndarray (12180, 4)
        track_test = np.array([track_test_mat['start_frame'], track_test_mat['end_frame'], 
                                track_test_mat['person_id'], track_test_mat['cam_id']]).squeeze(1).transpose(1,0)

        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        track_query = track_test[query_IDX,:]
        # use a set for fast lookup
        query_IDX_set = set(query_IDX.tolist() if hasattr(query_IDX, 'tolist') else query_IDX)
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX_set]
        track_gallery = track_test[gallery_IDX,:]
        # Build tracklets
        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_cams, num_train_vids = \
            self._process_data(train_names, track_train, home_dir='train', relabel=True,
                               min_seq_len=min_seq_len, json_path=self.split_train_json_path)

        query, num_query_tracklets, num_query_pids, num_query_imgs, query_pid, query_camid = \
            self._process_gallery_data(test_names, track_query, home_dir='test', relabel=False,
                                       min_seq_len=min_seq_len, json_path=self.split_query_json_path,)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, gallery_pid, gallery_camid = \
            self._process_gallery_data(test_names, track_gallery, home_dir='test', relabel=False,
                                       min_seq_len=min_seq_len, json_path=self.split_gallery_json_path)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        # Number of cameras in query and gallery
        num_query_cams = len(set(query_camid)) if query_camid else 0
        num_gallery_cams = len(set(gallery_camid)) if gallery_camid else 0

        print("=> EvReID loaded")
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # tracklets | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_cams))
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  ----------------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        self.queryinfo = infostruct()
        self.queryinfo.pid = query_pid
        self.queryinfo.camid = query_camid
        self.queryinfo.tranum = num_query_imgs

        self.galleryinfo = infostruct()
        self.galleryinfo.pid = gallery_pid
        self.galleryinfo.camid = gallery_camid
        self.galleryinfo.tranum = num_gallery_imgs

        self.num_train_cams = num_train_cams
        self.num_train_vids = num_train_vids
        self.num_query_cams = num_query_cams
        self.num_gallery_cams = num_gallery_cams
        
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))
    
    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names
    
    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, json_path=''):
        """
        Args:
            names: image file names ['0001C1T0001F001.jpg', ...]
            meta_data: tracklet info (N, 4) [start_frame, end_frame, pid, camid]
            home_dir: 'train' or 'test'
            relabel: relabel pids to 0..N-1 (training set)
            min_seq_len: minimum tracklet length
            json_path: JSON cache path
        """
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet'], split['num_cams'], split['num_tracks']
        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        assert home_dir in ['train', 'test']
        # First run: build the data
        num_tracklets = meta_data.shape[0]  # 8298
        pid_list = list(set(meta_data[:, 2].tolist()))  # unique pids
        num_pids = len(pid_list)
        # Relabel pids (training set: 625 ids -> 0..624)
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)}  # {1:0,3:1,5:2,...}
        tracklets = []
        num_imgs_per_tracklet = []
        cams = []
        # For each tracklet
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]  # [1 16 1 1]
            # e.g. [1, 16, 1, 1]: frames 1-16, pid 1, camera 1
            start_index, end_index, pid, camid = data
            
            cams += [int(camid)]
            
            if pid == -1:
                continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel:
                pid = pid2label[pid]  # pid = 0
            camid -= 1 # 0-based camera id
            # Image names of this tracklet
            img_names = names[start_index:end_index+1]
            # e.g. ['0001C1T0001F001.jpg', ..., '0001C1T0001F016.jpg']

            # Sanity checks
            pnames = [img_name[:4] for img_name in img_names]  # pnames = ['0001','0001'...]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]  # camnames = ['1','1'...]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # RGB path: '/path/to/rgb_degrade/train/0001/0001C1T0001F001.jpg'
            rgb_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]

            # Event path: '/path/to/event/train/0001/0001C1T0001F001.jpg'
            event_paths = [osp.join(self.root, home_dir, img_name[:4], img_name).replace('rgb_degrade', 'event').replace('rgb','event') for img_name in img_names]
            # Skip short tracklets
            if len(rgb_paths) >= min_seq_len:
                rgb_paths = tuple(rgb_paths)
                event_paths = tuple(event_paths)
                tracklets.append((rgb_paths, event_paths, int(pid), int(camid), 1))
                #                 RGB paths, Event paths, pid, camid, trackid
                num_imgs_per_tracklet.append(len(rgb_paths))


        num_tracklets = len(tracklets)  # 8298
        cams = set(cams)
        num_cams = len(cams)
        print("Saving split to {}".format(json_path))
        # Save to the JSON cache
        split_dict = {
            'tracklets': [list(item[:2]) + list(map(int, item[2:])) for item in tracklets],
            'num_tracklets': int(num_tracklets),
            'num_pids': int(num_pids),
            'num_imgs_per_tracklet': [int(x) for x in num_imgs_per_tracklet],
            'num_cams': int(num_cams),
            'num_tracks': 1
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_cams, 1
    
    def _process_gallery_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, json_path=''):
        # Differences from the training set:
        # 1. no relabelling (original pids)
        # 2. also returns the pid and camid of every tracklet

        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet'], split['pids'], split['camid']

        assert home_dir in ['train', 'test']
        num_tracklets = meta_data.shape[0]  # 8298
        pid_list = list(set(meta_data[:, 2].tolist()))  # pid = 625 => [1 3 5 7 9...]
        num_pids = len(pid_list)

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)}  # {1:0,3:1,5:2,...}
        tracklets = []
        num_imgs_per_tracklet = []
        gallery_pid = []    # pid of every tracklet
        gallery_camid = []  # camid of every tracklet

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]  # [1 16 1 1]
            start_index, end_index, pid, camid = data

            if pid == -1:
                continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel:
                pid = pid2label[pid]  # pid = 0
            camid -= 1
            # index starts from 0
            img_names = names[start_index:end_index+1]
            # <class 'list'>:['0001C1T0001F001.jpg'.. '0001C1T0001F016.jpg']

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]  # pnames = ['0001','0001'...]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]  # camnames = ['1','1'...]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            rgb_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]  # list<16>
            event_paths = [osp.join(self.root, home_dir, img_name[:4], img_name).replace('rgb_degrade', 'event').replace('rgb','event') for img_name in img_names]

            if len(rgb_paths) >= min_seq_len:
                rgb_paths = tuple(rgb_paths)
                event_paths = tuple(event_paths)
                tracklets.append((rgb_paths, event_paths, int(pid), int(camid), 1))  # (rgb_paths, event_paths, pid, camid, trackid)
                num_imgs_per_tracklet.append(len(rgb_paths))  # number of frames per tracklet
                gallery_pid.append(int(pid))
                gallery_camid.append(int(camid))
        num_tracklets = len(tracklets)  # 8298
        print("Saving split to {}".format(json_path))
        split_dict = {
            'tracklets': tracklets,
            'num_tracklets': num_tracklets,
            'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet,
            'pids': gallery_pid,
            'camid': gallery_camid,
        }
        write_json(split_dict, json_path)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, gallery_pid, gallery_camid

if __name__ == '__main__':
    # test
    dataset = EvReID()

