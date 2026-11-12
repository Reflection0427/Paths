from __future__ import print_function, absolute_import
import os
import glob
import sys
import errno
import json
import os.path as osp
import numpy as np
import urllib
import tarfile
from scipy.io import loadmat


# File helpers
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


# Container for extra query/gallery info
class infostruct(object):
    pass


class iLIDSVID(object):
    """
    iLIDS-VID RGB-Event dataset.
    Builds paired RGB/Event tracklets and caches the splits as JSON. Tracklet format: (rgb_paths, event_paths, pid, camid, viewid).
    """
    # Default paths (overridden by the config)
    root = 'datasets/i-LIDS-VID'
    event_root = 'datasets/i-LIDS-VID/i-LIDS-VID_mat_divide_jpg'
    dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/i-LIDS-VID/i-LIDS-VID.tar'

    def __init__(self, root=None, event_root=None, degrade='', split_id=0, min_seq_len=0, **kwargs):
        """
        Args:
            root: dataset root containing i-LIDS-VID[_<degrade>]/ and the split files
            event_root: root of the event frames
            degrade: '' (clean), 'block' (occlusion), 'blur' or 'dark'
            split_id: which of the 10 train/test splits to use (0-9)
            min_seq_len: drop tracklets shorter than this
        """
        if root:
            self.root = root
        if event_root:
            self.event_root = event_root
        rgb_dir = 'i-LIDS-VID' + (f'_{degrade}' if degrade else '')
        self.split_path = osp.join(self.root, 'splits.json')
        self.split_mat_path = osp.join(self.root, 'train-test people splits', 'train_test_splits_ilidsvid.mat')
        self.rgb_cam1_path = osp.join(self.root, rgb_dir, 'sequences', 'cam1')
        self.rgb_cam2_path = osp.join(self.root, rgb_dir, 'sequences', 'cam2')
        self.event_cam1_path = osp.join(self.event_root, 'sequences', 'cam1')
        self.event_cam2_path = osp.join(self.event_root, 'sequences', 'cam2')
        # JSON cache files (one set per degradation)
        json_suffix = f"_{degrade}" if degrade else ""
        self.split_train_json_path = osp.join(self.root, f'cache_train{json_suffix}.json')
        self.split_query_json_path = osp.join(self.root, f'cache_query{json_suffix}.json')
        self.split_gallery_json_path = osp.join(self.root, f'cache_gallery{json_suffix}.json')

        # Check dataset paths
        self._check_before_run()

        # Create the split file if needed
        self._prepare_split()

        # Load the split file
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                "split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits) - 1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        # Build train / query / gallery
        train, num_train_tracklets, num_train_pids, num_imgs_train, num_train_cams, _ = \
            self._process_data(train_dirs, cam1=True, cam2=True, home_dir='train',
                               relabel=True, min_seq_len=min_seq_len,
                               json_path=self.split_train_json_path)

        query, num_query_tracklets, num_query_pids, num_imgs_query, query_pid, query_camid = \
            self._process_gallery_data(test_dirs, cam1=True, cam2=False, home_dir='test',
                                       relabel=False, min_seq_len=min_seq_len,
                                       json_path=self.split_query_json_path)

        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery, gallery_pid, gallery_camid = \
            self._process_gallery_data(test_dirs, cam1=False, cam2=True, home_dir='test',
                                       relabel=False, min_seq_len=min_seq_len,
                                       json_path=self.split_gallery_json_path)

        # Global statistics
        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet) if num_imgs_per_tracklet else 0
        max_num = np.max(num_imgs_per_tracklet) if num_imgs_per_tracklet else 0
        avg_num = np.mean(num_imgs_per_tracklet) if num_imgs_per_tracklet else 0

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        # Print dataset statistics
        print("=> iLIDS-VID (RGB+EVENT) loaded")
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # tracklets | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_tracklets, len(set(query_camid))))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets,
                                                          len(set(gallery_camid))))
        print("  ----------------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ----------------------------------------")

        # Dataset attributes
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        # Extra query/gallery info
        self.queryinfo = infostruct()
        self.queryinfo.pid = query_pid
        self.queryinfo.camid = query_camid
        self.queryinfo.tranum = num_imgs_query

        self.galleryinfo = infostruct()
        self.galleryinfo.pid = gallery_pid
        self.galleryinfo.camid = gallery_camid
        self.galleryinfo.tranum = num_imgs_gallery

        self.num_train_cams = num_train_cams
        self.num_query_cams = len(set(query_camid)) if query_camid else 0
        self.num_gallery_cams = len(set(gallery_camid)) if gallery_camid else 0

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        # Check RGB paths
        if not osp.exists(self.root):
            raise RuntimeError(f"RGB Dataset root not found: '{self.root}'")
        if not osp.exists(self.rgb_cam1_path):
            raise RuntimeError(f"RGB Camera 1 path not found: '{self.rgb_cam1_path}'")
        if not osp.exists(self.rgb_cam2_path):
            raise RuntimeError(f"RGB Camera 2 path not found: '{self.rgb_cam2_path}'")

        # Check Event paths
        if not osp.exists(self.event_root):
            raise RuntimeError(f"EVENT Dataset root not found: '{self.event_root}'")
        if not osp.exists(self.event_cam1_path):
            raise RuntimeError(f"EVENT Camera 1 path not found: '{self.event_cam1_path}'")
        if not osp.exists(self.event_cam2_path):
            raise RuntimeError(f"EVENT Camera 2 path not found: '{self.event_cam2_path}'")

        # Check the split .mat file
        if not osp.exists(self.split_mat_path):
            raise RuntimeError(f"Split mat file not found: '{self.split_mat_path}'")

    def _prepare_split(self):
        """Create splits.json from the official .mat split file."""
        # Skip if it already exists
        if osp.exists(self.split_path):
            print(f"=> {self.split_path} generated before, loading from cache!")
            return

        print(f"=> Generating {self.split_path} (first run, might take a while)")

        # Load the .mat split file
        mat_split_data = loadmat(self.split_mat_path)['ls_set']
        num_splits = mat_split_data.shape[0]
        num_total_ids = mat_split_data.shape[1]
        print(f"Note: Found {num_total_ids} IDs in mat file (expected 300 for standard iLIDS-VID)")

        # Check the number of splits
        assert num_splits == 10, f"Expected 10 splits, got {num_splits}"
        num_ids_each = num_total_ids // 2

        # Identity folder names
        person_cam1_dirs = sorted(glob.glob(osp.join(self.rgb_cam1_path, '*')))
        person_cam2_dirs = sorted(glob.glob(osp.join(self.rgb_cam2_path, '*')))
        person_cam1_dirs = [osp.basename(item) for item in person_cam1_dirs]
        person_cam2_dirs = [osp.basename(item) for item in person_cam2_dirs]

        # Keep identities present in both cam1 and cam2
        assert set(person_cam1_dirs) == set(person_cam2_dirs), "Cam1 and Cam2 have different persons"

        # Valid identities
        actual_num_ids = len(person_cam1_dirs)
        valid_total_ids = min(num_total_ids, actual_num_ids)
        if valid_total_ids < num_total_ids:
            print(f"Warning: Truncating mat IDs from {num_total_ids} to {valid_total_ids} to match actual directories")
            num_total_ids = valid_total_ids
            num_ids_each = num_total_ids // 2

        # Build splits
        splits = []
        for i_split in range(num_splits):
            all_idxs = sorted(list(mat_split_data[i_split, :num_total_ids]))
            all_idxs = [int(i) - 1 for i in all_idxs]  # to 0-based indices
            all_idxs = [i for i in all_idxs if i < actual_num_ids]  # drop out-of-range indices

            # half train / half test
            train_idxs = all_idxs[num_ids_each:]
            test_idxs = all_idxs[:num_ids_each]

            # indices -> folder names
            train_dirs = [person_cam1_dirs[i] for i in train_idxs]
            test_dirs = [person_cam1_dirs[i] for i in test_idxs]

            splits.append({'train': train_dirs, 'test': test_dirs})

        # Save splits.json
        write_json(splits, self.split_path)
        print(f"Totally {len(splits)} splits created (each with {len(train_dirs)} train / {len(test_dirs)} test IDs)")

    def _process_data(self, dirnames, cam1=True, cam2=True, home_dir=None, relabel=False, min_seq_len=0, json_path=''):
        """
        Build tracklets for the given identity folders.
        Returns: (tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_cams, 1)
        Tracklet: (rgb_paths, event_paths, pid, camid, viewid)
        """
        # Load from the JSON cache if present
        if osp.exists(json_path):
            print(f"=> {json_path} generated before, loading from cache!")
            split = read_json(json_path)
            # convert back to tuples
            tracklets = [
                (tuple(item[0]), tuple(item[1]), item[2], item[3], item[4])
                for item in split['tracklets']
            ]
            return (tracklets, split['num_tracklets'], split['num_pids'],
                    split['num_imgs_per_tracklet'], split['num_cams'], 1)

        print(f"=> Generating {json_path} (first run, might take a while)")

        # Init
        tracklets = []
        num_imgs_per_tracklet = []
        cams = []

        # Integer PIDs
        pid_list = []
        for dirname in dirnames:
            try:
                pid_int = int(dirname.replace('person', ''))
            except:
                pid_int = hash(dirname) % 10000  # fallback for non-numeric names
            pid_list.append(pid_int)
        num_pids = len(pid_list)

        # Relabel PIDs to 0..N-1 (training set)
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)}

        # For each identity
        for dirname in dirnames:
            # cam1
            if cam1:
                # RGB paths
                rgb_person_dir = osp.join(self.rgb_cam1_path, dirname)
                rgb_img_names = sorted(glob.glob(osp.join(rgb_person_dir, '*.png')))
                # Event paths
                event_person_dir = osp.join(self.event_cam1_path, dirname)
                event_img_names = sorted(glob.glob(osp.join(event_person_dir, '*.png')))

                # skip empty tracklets
                if len(rgb_img_names) == 0 or len(event_img_names) == 0:
                    print(f"Warning: {dirname} has no RGB/Event frames in cam1, skipped")
                    continue

                # keep the same number of RGB and Event frames
                min_len = min(len(rgb_img_names), len(event_img_names))
                rgb_img_names = rgb_img_names[:min_len]
                event_img_names = event_img_names[:min_len]

                # skip short tracklets
                if len(rgb_img_names) < min_seq_len:
                    continue

                # PID and camera ID
                try:
                    pid_int = int(dirname.replace('person', ''))
                except:
                    pid_int = hash(dirname) % 10000
                pid = pid_int if not relabel else pid2label[pid_int]
                camid = 0  # cam1 = 0
                viewid = 1  # single view

                # tracklet tuple
                rgb_paths = tuple(rgb_img_names)
                event_paths = tuple(event_img_names)
                tracklets.append((rgb_paths, event_paths, pid, camid, viewid))
                num_imgs_per_tracklet.append(len(rgb_paths))
                cams.append(camid)

            # cam2
            if cam2:
                # RGB paths
                rgb_person_dir = osp.join(self.rgb_cam2_path, dirname)
                rgb_img_names = sorted(glob.glob(osp.join(rgb_person_dir, '*.png')))
                # Event paths
                event_person_dir = osp.join(self.event_cam2_path, dirname)
                event_img_names = sorted(glob.glob(osp.join(event_person_dir, '*.png')))

                # skip empty tracklets
                if len(rgb_img_names) == 0 or len(event_img_names) == 0:
                    print(f"Warning: {dirname} has no RGB/Event frames in cam2, skipped")
                    continue

                # keep the same number of RGB and Event frames
                min_len = min(len(rgb_img_names), len(event_img_names))
                rgb_img_names = rgb_img_names[:min_len]
                event_img_names = event_img_names[:min_len]

                # skip short tracklets
                if len(rgb_img_names) < min_seq_len:
                    continue

                # PID and camera ID
                try:
                    pid_int = int(dirname.replace('person', ''))
                except:
                    pid_int = hash(dirname) % 10000
                pid = pid_int if not relabel else pid2label[pid_int]
                camid = 1  # cam2 = 1
                viewid = 1

                # tracklet tuple
                rgb_paths = tuple(rgb_img_names)
                event_paths = tuple(event_img_names)
                tracklets.append((rgb_paths, event_paths, pid, camid, viewid))
                num_imgs_per_tracklet.append(len(rgb_paths))
                cams.append(camid)

        # Statistics
        num_tracklets = len(tracklets)
        num_cams = len(set(cams)) if cams else 0

        # Save the JSON cache
        split_dict = {
            'tracklets': [list(item[:2]) + [int(x) for x in item[2:]] for item in tracklets],
            'num_tracklets': int(num_tracklets),
            'num_pids': int(num_pids),
            'num_imgs_per_tracklet': [int(x) for x in num_imgs_per_tracklet],
            'num_cams': int(num_cams),
            'num_tracks': 1
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_cams, 1

    def _process_gallery_data(self, dirnames, cam1=True, cam2=True, home_dir=None, relabel=False, min_seq_len=0,
                              json_path=''):
        """
        Build query / gallery tracklets.
        Returns: (tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, pids, camids)
        """
        # Load from the JSON cache if present
        if osp.exists(json_path):
            print(f"=> {json_path} generated before, loading from cache!")
            split = read_json(json_path)
            # convert back to tuples
            tracklets = [
                (tuple(item[0]), tuple(item[1]), item[2], item[3], item[4])
                for item in split['tracklets']
            ]
            return (tracklets, split['num_tracklets'], split['num_pids'],
                    split['num_imgs_per_tracklet'], split['pids'], split['camid'])

        # Init
        tracklets = []
        num_imgs_per_tracklet = []
        gallery_pid = []
        gallery_camid = []

        # Integer PIDs
        pid_list = []
        for dirname in dirnames:
            try:
                pid_int = int(dirname.replace('person', ''))
            except:
                pid_int = hash(dirname) % 10000
            pid_list.append(pid_int)
        num_pids = len(pid_list)

        # Relabel PIDs (training set only)
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)}

        # For each identity
        for dirname in dirnames:
            # cam1
            if cam1:
                rgb_person_dir = osp.join(self.rgb_cam1_path, dirname)
                rgb_img_names = sorted(glob.glob(osp.join(rgb_person_dir, '*.png')))
                event_person_dir = osp.join(self.event_cam1_path, dirname)
                event_img_names = sorted(glob.glob(osp.join(event_person_dir, '*.png')))

                if len(rgb_img_names) == 0 or len(event_img_names) == 0:
                    print(f"Warning: {dirname} has no RGB/Event frames in cam1, skipped")
                    continue

                min_len = min(len(rgb_img_names), len(event_img_names))
                rgb_img_names = rgb_img_names[:min_len]
                event_img_names = event_img_names[:min_len]

                if len(rgb_img_names) < min_seq_len:
                    continue

                # PID and camera ID
                try:
                    pid_int = int(dirname.replace('person', ''))
                except:
                    pid_int = hash(dirname) % 10000
                pid = pid_int if not relabel else pid2label[pid_int]
                camid = 0
                viewid = 1

                # tracklet tuple
                rgb_paths = tuple(rgb_img_names)
                event_paths = tuple(event_img_names)
                tracklets.append((rgb_paths, event_paths, pid, camid, viewid))
                num_imgs_per_tracklet.append(len(rgb_paths))
                gallery_pid.append(pid)
                gallery_camid.append(camid)

            # cam2
            if cam2:
                rgb_person_dir = osp.join(self.rgb_cam2_path, dirname)
                rgb_img_names = sorted(glob.glob(osp.join(rgb_person_dir, '*.png')))
                event_person_dir = osp.join(self.event_cam2_path, dirname)
                event_img_names = sorted(glob.glob(osp.join(event_person_dir, '*.png')))

                if len(rgb_img_names) == 0 or len(event_img_names) == 0:
                    print(f"Warning: {dirname} has no RGB/Event frames in cam2, skipped")
                    continue

                min_len = min(len(rgb_img_names), len(event_img_names))
                rgb_img_names = rgb_img_names[:min_len]
                event_img_names = event_img_names[:min_len]

                if len(rgb_img_names) < min_seq_len:
                    continue

                # PID and camera ID
                try:
                    pid_int = int(dirname.replace('person', ''))
                except:
                    pid_int = hash(dirname) % 10000
                pid = pid_int if not relabel else pid2label[pid_int]
                camid = 1
                viewid = 1

                # tracklet tuple
                rgb_paths = tuple(rgb_img_names)
                event_paths = tuple(event_img_names)
                tracklets.append((rgb_paths, event_paths, pid, camid, viewid))
                num_imgs_per_tracklet.append(len(rgb_paths))
                gallery_pid.append(pid)
                gallery_camid.append(camid)

        # Statistics
        num_tracklets = len(tracklets)

        # Save the JSON cache
        split_dict = {
            'tracklets': [list(item[:2]) + [int(x) for x in item[2:]] for item in tracklets],
            'num_tracklets': int(num_tracklets),
            'num_pids': int(num_pids),
            'num_imgs_per_tracklet': [int(x) for x in num_imgs_per_tracklet],
            'pids': [int(p) for p in gallery_pid],
            'camid': [int(c) for c in gallery_camid],
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, gallery_pid, gallery_camid
