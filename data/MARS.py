"""
MARS Dataset Loader for Multi-modal Person Re-identification

MARS (Motion Analysis and Re-identification Set), a large-scale video person ReID dataset.
This loader reads paired RGB and Event frames.
"""

from __future__ import print_function, absolute_import
import json
import os.path as osp
import numpy as np
from scipy.io import loadmat

# from utils.serialization import write_json, read_json

# ==================== Constants ====================
NUM_CAMERAS = 6  # number of cameras in MARS
INVALID_PID = -1  # marks invalid identities
# ==================== Helpers ====================
class InfoStruct:
    """Metadata of the query / gallery sets."""
    pass

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


# ==================== Dataset ====================
class MARS:
    """
    MARS RGB-Event dataset.

    Layout:
        mars/
        ├── rgb/
        │   ├── bbox_train/    # training RGB frames
        │   └── bbox_test/     # test RGB frames
        ├── event/             # Event frames (same layout as rgb)
        └── info/              # metadata
            ├── train_name.txt
            ├── test_name.txt
            ├── tracks_train_info.mat
            ├── tracks_test_info.mat
            └── query_IDX.mat

    Attributes:
        train: training tracklets
        query: query tracklets
        gallery: gallery tracklets
        num_train_pids: number of training identities
        num_query_pids: number of query identities
        num_gallery_pids: number of gallery identities
        queryinfo: query metadata
        galleryinfo: gallery metadata
    """

    # default root (overridden by the config)
    root = 'datasets/MARS'

    def __init__(self, root='datasets/MARS', min_seq_len=0):
        """
        Load MARS.

        Args:
            root: dataset root
            min_seq_len: drop tracklets shorter than this
        """
        # root
        if root:
            self.root = root

        # file paths
        self._setup_paths()
        # check required files
        self._check_before_run()
        # load and process
        self._load_and_process_data(min_seq_len)
        # Print dataset statistics.
        self._print_dataset_statistics()

    def _setup_paths(self):
        """Set all file paths."""
        # original metadata
        self.train_name_path = osp.join(self.root, 'info/train_name.txt')
        self.test_name_path = osp.join(self.root, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self.root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self.root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self.root, 'info/query_IDX.mat')

        # JSON cache path
        self.split_train_json_path = osp.join(self.root, 'split_train.json')
        self.split_query_json_path = osp.join(self.root, 'split_query.json')
        self.split_gallery_json_path = osp.join(self.root, 'split_gallery.json')

    def _check_before_run(self):
        """Check that all required files exist."""
        required_paths = [
            self.root,
            self.train_name_path,
            self.test_name_path,
            self.track_train_info_path,
            self.track_test_info_path,
            self.query_IDX_path
        ]

        for path in required_paths:
            if not osp.exists(path):
                raise RuntimeError(f"'{path}' is not available")

    def _load_and_process_data(self, min_seq_len):
        """Load and process all splits."""
        # 1. image names
        train_names = self._load_image_names(self.train_name_path)
        test_names = self._load_image_names(self.test_name_path)

        # 2. tracklet metadata
        track_train = self._load_track_info(self.track_train_info_path, 'track_train_info')
        track_test = self._load_track_info(self.track_test_info_path, 'track_test_info')

        # 3. split test into query / gallery
        track_query, track_gallery = self._split_query_gallery(track_test)

        # 4. training set
        train_result = self._process_data(
            names=train_names,
            meta_data=track_train,
            home_dir='rgb/bbox_train',
            relabel=True,
            min_seq_len=min_seq_len,
            json_path=self.split_train_json_path
        )
        self.train, self.num_train_pids, self.num_train_cams, self.num_train_vids, num_train_imgs = train_result

        # 5. query set
        query_result = self._process_gallery_data(
            names=test_names,
            meta_data=track_query,
            home_dir='rgb/bbox_test',
            relabel=False,
            min_seq_len=min_seq_len,
            json_path=self.split_query_json_path
        )
        self.query, self.num_query_pids, self.queryinfo, num_query_imgs = query_result

        # 6. gallery set
        gallery_result = self._process_gallery_data(
            names=test_names,
            meta_data=track_gallery,
            home_dir='rgb/bbox_test',
            relabel=False,
            min_seq_len=min_seq_len,
            json_path=self.split_gallery_json_path
        )
        self.gallery, self.num_gallery_pids, self.galleryinfo, num_gallery_imgs = gallery_result

        # statistics for printing
        self._save_statistics(num_train_imgs, num_query_imgs, num_gallery_imgs)

    def _load_image_names(self, file_path):
        """
        Load image names from a text file.

        Args:
            file_path: file path

        Returns:
            list of image names
        """
        names = []
        with open(file_path, 'r') as f:
            for line in f:
                names.append(line.rstrip())
        return names

    def _load_track_info(self, file_path, key):
        """
        Load tracklet info from a .mat file.

        Args:
            file_path: .mat file path
            key: key in the .mat dict

        Returns:
            numpy.ndarray: shape (N, 4), rows are [start_frame, end_frame, person_id, camera_id]
        """
        return loadmat(file_path)[key]

    def _split_query_gallery(self, track_test):
        """
        Split the test set into query and gallery.

        Args:
            track_test: test tracklet info

        Returns:
            track_query: query tracklet info
            track_gallery: gallery tracklet info
        """
        # query indices (1-based MATLAB indices)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()
        query_IDX -= 1  # to 0-based

        # query
        track_query = track_test[query_IDX, :]

        # gallery = test set minus query
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]

        return track_query, track_gallery

    def _process_data(self, names, meta_data, home_dir, relabel, min_seq_len, json_path):
        """
        Build training tracklets.

        Args:
            names: list of image names
            meta_data: tracklet metadata (N, 4)
            home_dir: image folder
            relabel: relabel person ids to 0..N-1 (training set)
            min_seq_len: minimum tracklet length
            json_path: JSON cache path

        Returns:
            tracklets: tracklets
            num_pids: number of identities
            num_cams: number of cameras
            num_tracks: number of tracks
            num_imgs_per_tracklet: frames per tracklet
        """
        # use the cache if present
        if osp.exists(json_path):
            print(f"=> {json_path} generated before, loading from cache")
            split = read_json(json_path)
            return (split['tracklets'], split['num_pids'], split['num_cams'],
                    split['num_tracks'], split['num_imgs_per_tracklet'])

        print("=> Generating split for the first time (this might take a while)...")

        # unique person ids
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        # person id -> label (when relabelling)
        pid2label = None
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)}

        # each tracklet
        tracklets = []
        num_imgs_per_tracklet = []
        camera_ids = []

        for tracklet_idx in range(num_tracklets):
            tracklet_data = self._process_single_tracklet(
                meta_data[tracklet_idx],
                names,
                home_dir,
                pid2label,
                min_seq_len
            )

            if tracklet_data is not None:
                tracklet, num_imgs, cam_id = tracklet_data
                tracklets.append(tracklet)
                num_imgs_per_tracklet.append(num_imgs)
                camera_ids.append(cam_id)

        num_cams = len(set(camera_ids))

        # save to the JSON cache
        self._save_split_to_json(
            json_path, tracklets, num_pids, num_imgs_per_tracklet, num_cams, 1
        )

        return tracklets, num_pids, num_cams, 1, num_imgs_per_tracklet

    def _process_single_tracklet(self, track_info, names, home_dir, pid2label, min_seq_len):
        """
        Process one tracklet.

        Args:
            track_info: [start_frame, end_frame, person_id, camera_id]
            names: list of image names
            home_dir: image folder
            pid2label: person id -> label (None: keep original ids)
            min_seq_len: minimum tracklet length

        Returns:
            (tracklet_tuple, num_imgs, camera_id) for a valid tracklet
            None for an invalid tracklet
        """
        start_index, end_index, pid, camid = track_info

        # skip invalid entries
        if pid == INVALID_PID:
            return None

        # check camera id
        assert 1 <= camid <= NUM_CAMERAS, f"Invalid camera ID: {camid}"

        # relabel if needed
        if pid2label is not None:
            pid = pid2label[pid]

        # 0-based camera id
        camid = int(camid) - 1

        # image names (MATLAB -> Python indices)
        img_names = names[int(start_index) - 1:int(end_index)]

        # consistency checks
        self._validate_tracklet(img_names)

        # image paths
        img_paths_rgb = self._build_image_paths(img_names, home_dir, 'rgb')
        img_paths_event = self._build_image_paths(img_names, home_dir, 'event')

        # length check
        if len(img_paths_rgb) < min_seq_len:
            return None

        # tracklet data
        tracklet = (
            tuple(img_paths_rgb),
            tuple(img_paths_event),
            int(pid),
            int(camid),
            1  # track_id
        )

        return tracklet, len(img_paths_rgb), int(camid) + 1

    def _validate_tracklet(self, img_names):
        """
        Check that a tracklet is consistent.

        Args:
            img_names: list of image names
        """
        # all frames belong to the same identity
        person_ids = [img_name[:4] for img_name in img_names]
        assert len(set(person_ids)) == 1, "Tracklet contains images from different persons"

        # all frames come from the same camera
        camera_ids = [img_name[5] for img_name in img_names]
        assert len(set(camera_ids)) == 1, "Tracklet contains images from different cameras"

    def _build_image_paths(self, img_names, home_dir, modality):
        """
        Build full image paths.

        Args:
            img_names: list of image names
            home_dir: image folder (relative to root)
            modality: modality ('rgb' or 'event')

        Returns:
            full image paths
        """
        # swap the modality folder
        if modality == 'event':
            home_dir = home_dir.replace('rgb', 'event')

        img_paths = [
            osp.join(self.root, home_dir, img_name[:4], img_name)
            for img_name in img_names
        ]

        return img_paths

    def _process_gallery_data(self, names, meta_data, home_dir, relabel, min_seq_len, json_path):
        """
        Build query / gallery tracklets.

        Differences from _process_data:
        1. no relabelling (original person ids)
        2. also records pid and camid of every tracklet (for evaluation)

        Returns:
            tracklets: tracklets
            num_pids: number of identities
            info_struct: struct with pid, camid and tranum
            num_imgs_per_tracklet: frames per tracklet
        """
        # use the cache if present
        if osp.exists(json_path):
            print(f"=> {json_path} generated before, loading from cache")
            split = read_json(json_path)

            # info struct
            info = InfoStruct()
            info.pid = split['pids']
            info.camid = split['camid']
            info.tranum = split['num_imgs_per_tracklet']

            return split['tracklets'], split['num_pids'], info, split['num_imgs_per_tracklet']

        print("=> Generating split for the first time...")

        # unique person ids
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        # each tracklet
        tracklets = []
        num_imgs_per_tracklet = []
        gallery_pids = []
        gallery_camids = []

        for tracklet_idx in range(num_tracklets):
            tracklet_data = self._process_single_tracklet(
                meta_data[tracklet_idx],
                names,
                home_dir,
                pid2label=None,  # no relabelling
                min_seq_len=min_seq_len
            )

            # original pid and camid
            _, _, pid, camid = meta_data[tracklet_idx]

            if tracklet_data is not None:
                tracklet, num_imgs, _ = tracklet_data
                tracklets.append(tracklet)
                num_imgs_per_tracklet.append(num_imgs)

            # record pid and camid of every tracklet (including filtered ones)
            gallery_pids.append(int(pid))
            gallery_camids.append(int(camid) - 1)

        # save to the JSON cache
        split_dict = {
            'tracklets': tracklets,
            'num_tracklets': len(tracklets),
            'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet,
            'pids': gallery_pids,
            'camid': gallery_camids
        }
        write_json(split_dict, json_path)

        # info struct
        info = InfoStruct()
        info.pid = gallery_pids
        info.camid = gallery_camids
        info.tranum = num_imgs_per_tracklet

        return tracklets, num_pids, info, num_imgs_per_tracklet

    def _save_split_to_json(self, json_path, tracklets, num_pids, num_imgs_per_tracklet, num_cams, num_tracks):
        """Save a split to JSON."""
        print(f"=> Saving split to {json_path}")

        split_dict = {
            'tracklets': tracklets,
            'num_tracklets': len(tracklets),
            'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet,
            'num_cams': num_cams,
            'num_tracks': num_tracks
        }
        write_json(split_dict, json_path)

    def _save_statistics(self, num_train_imgs, num_query_imgs, num_gallery_imgs):
        """Store statistics."""
        # all frame counts
        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs

        self.min_num_imgs = np.min(num_imgs_per_tracklet)
        self.max_num_imgs = np.max(num_imgs_per_tracklet)
        self.avg_num_imgs = np.mean(num_imgs_per_tracklet)

        self.num_train_tracklets = len(self.train)
        self.num_query_tracklets = len(self.query)
        self.num_gallery_tracklets = len(self.gallery)

        self.num_total_pids = self.num_train_pids + self.num_query_pids
        self.num_total_tracklets = (self.num_train_tracklets +
                                    self.num_query_tracklets +
                                    self.num_gallery_tracklets)

    def _print_dataset_statistics(self):
        """Print dataset statistics."""
        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print(f"  train    | {self.num_train_pids:5d} | {self.num_train_tracklets:8d}")
        print(f"  query    | {self.num_query_pids:5d} | {self.num_query_tracklets:8d}")
        print(f"  gallery  | {self.num_gallery_pids:5d} | {self.num_gallery_tracklets:8d}")
        print("  ------------------------------")
        print(f"  total    | {self.num_total_pids:5d} | {self.num_total_tracklets:8d}")
        print("  ------------------------------")
        print(f"  number of images per tracklet: {self.min_num_imgs} ~ {self.max_num_imgs}, "
              f"average {self.avg_num_imgs:.1f}")
        print("  ------------------------------")
