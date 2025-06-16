import os
import math
import glob
import h5py
import shutil
import random

import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset, Subset

seed = 42
random.seed(seed)
prng = np.random.RandomState(seed)


class IconColumnIterableDataset(IterableDataset):
    """An IterableDataset class to load data from disk"""

    def __init__(
        self,
        files_path,
        temporal_subsample=None,
        spatial_subsample=None,
        cache_dir=None,
        shuffle=True,
        transform=None,
    ):
        super(IconColumnIterableDataset).__init__()
        self.spatial_subsample = spatial_subsample
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.transform = transform
        if temporal_subsample is not None and temporal_subsample < 1.0:
            self.files_path = prng.choice(
                files_path, max(1, int(temporal_subsample * len(files_path)))
            )
        else:
            self.files_path = files_path

    def read_h5(self, file):
        try:
            with h5py.File(file, "r") as h:
                x3d = torch.tensor(h["x3d"][:], dtype=torch.float32)
                x2d = torch.tensor(h["x2d"][:], dtype=torch.float32)
                y = torch.tensor(h["y"][:], dtype=torch.float32)
        except:
            raise Exception(f"Error in reading :{file}")

        return x3d, x2d, y

    def load_files(self, remote_file):
        try:
            local_file = os.path.join(self.cache_dir, os.path.basename(remote_file))
            x3d, x2d, y = self.read_h5(local_file)
        except:
            x3d, x2d, y = self.read_h5(remote_file)
            if self.cache_dir:
                shutil.copy(remote_file, local_file)

        if self.transform:
            x3d, x2d, y = self.transform((x3d, x2d, y))

        if self.spatial_subsample:
            idx = torch.randint(
                0, y.size(0), (int(self.spatial_subsample * y.size(0)),)
            )
            return x3d[idx], x2d[idx], y[idx]
        return x3d, x2d, y

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.files_path)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self.files_path)
        else:  # in a worker process
            per_worker = int(
                math.ceil(len(self.files_path) / float(worker_info.num_workers))
            )
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files_path))

        for file_path in self.files_path[iter_start:iter_end]:
            x3d, x2d, y = self.load_files(file_path)
            for x3d_, x2d_, y_ in zip(x3d, x2d, y):
                yield x3d_, x2d_, y_


class IconDataset(object):
    """An IterableDataset class to load data from disk"""

    def __init__(
        self,
        data_dir,
        temporal_subsample=None,
        spatial_subsample=None,
        cache_dir=None,
        shuffle=True,
        transform=None,
    ):
        super(IconDataset).__init__()
        self.files_path = sorted(glob.glob(os.path.join(data_dir, "ml*.h5")))

        self.data_dir = data_dir
        self.temporal_subsample = temporal_subsample
        self.spatial_subsample = spatial_subsample
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.transform = transform

    def get_split(self, split):
        return IconColumnIterableDataset(
            split,
            temporal_subsample=self.temporal_subsample,
            spatial_subsample=self.spatial_subsample,
            cache_dir=self.cache_dir,
            shuffle=self.shuffle,
            transform=self.transform,
        )

    def train(self):
        return self.get_split(self.files_path[200:2000])

    def val(self):
        return self.get_split(self.files_path[:160] + self.files_path[2020:2180])

    def test(self):
        return self.get_split(self.files_path[2220:])

    def split(self):
        return self.train(), self.val(), self.test()


class IconH5Dataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()

        self.h5_file = h5py.File(data_path, "r")
        self.x3d = self.h5_file["x3d"]
        self.x2d = self.h5_file["x2d"]
        self.y = self.h5_file["y"]
        self.t = self.x3d.shape[0]
        self.s = self.x3d.shape[1]

    def __len__(self):
        return self.t * self.s

    def __getitem__(self, idx):
        t, s = idx // self.s, idx % self.s
        x3d = self.x3d[t, s]
        x2d = self.x2d[t, s]
        y = self.y[t, s]
        return x3d, x2d, y

    def split(self):
        train_range = range(self.s * 200, self.s * 2000)
        val_range = list(range(self.s * 160)) + list(
            range(self.s * 2020, self.s * 2180)
        )
        test_range = range(self.s * 2220, len(self))

        return (
            Subset(self, train_range),
            Subset(self, val_range),
            Subset(self, test_range),
        )


class IconH5Metadata:
    def __init__(self, metadata_path) -> None:
        super().__init__()

        h5_file = h5py.File(metadata_path, "r")
        self.x3d_mean_pfph = torch.from_numpy(
            h5_file["x3d_mean_pfph"][:].astype(np.float32)
        )
        self.x3d_std_pfph = torch.from_numpy(
            h5_file["x3d_std_pfph"][:].astype(np.float32)
        )
        self.x3d_mean_pf = torch.from_numpy(
            h5_file["x3d_mean_pf"][:].astype(np.float32)
        )
        self.x3d_std_pf = torch.from_numpy(h5_file["x3d_std_pf"][:].astype(np.float32))
        self.x2d_mean = torch.from_numpy(h5_file["x2d_mean"][:].astype(np.float32))
        self.x2d_std = torch.from_numpy(h5_file["x2d_std"][:].astype(np.float32))
        self.y_mean_pfph = torch.from_numpy(
            h5_file["y_mean_pfph"][:].astype(np.float32)
        )
        self.y_std_pfph = torch.from_numpy(h5_file["y_std_pfph"][:].astype(np.float32))
        self.y_mean_pf = torch.from_numpy(h5_file["y_mean_pf"][:].astype(np.float32))
        self.y_std_pf = torch.from_numpy(h5_file["y_std_pf"][:].astype(np.float32))
        self.beta = torch.from_numpy(h5_file["beta"][:].astype(np.float32))

    @property
    def mean_std_pfph(self):
        return self.x3d_mean_pfph, self.x3d_std_pfph, self.x2d_mean, self.x2d_std

    @property
    def mean_std_pf(self):
        return self.x3d_mean_pf, self.x3d_std_pf, self.x2d_mean, self.x2d_std

    @property
    def mean_std_hybrid(self):
        return self.x3d_mean_pfph, self.x3d_std_pf, self.x2d_mean, self.x2d_std

    @property
    def beta_constant(self):
        return self.beta

    def get_mean_std(self, normalization):
        if normalization == "pfph":
            return self.mean_std_pfph
        if normalization == "pf":
            return self.mean_std_pf
        if normalization == "hybrid":
            return self.mean_std_hybrid


class OutputNormalizer:
    def __init__(self, y_mean, y_std) -> None:
        self.y_mean = y_mean
        self.y_std = y_std
        super().__init__()

    def __call__(self, data):
        x3d, x2d, y = data
        y_norm = (y - self.y_mean) / (self.y_std + 1e-10)
        return x3d, x2d, y_norm


class HeightCutter:
    def __init__(self, height_in, height_out):
        self.height_in = height_in
        self.height_out = height_out
        super().__init__()

    def __call__(self, data):
        x3d, x2d, y = data
        return x3d[:, : self.height_in, :], x2d, y[:, : self.height_out, :]
