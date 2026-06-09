import torch
import numpy as np
import os
from torch.utils.data import Dataset, ConcatDataset
from .dataset_with_path import ImageFolderWithPath
from glob import glob
import h5py
from tqdm import tqdm
from collections import defaultdict
import random
import torch.distributed as dist

class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = None
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["code"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
        code = torch.from_numpy(self.file["code"][idx])
        label = int(self.file["label"][idx])
        path = self.file["path"][idx].decode("utf-8")
        return code, label, path


class H5DatasetInMemory(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            # 一次性全部加载到内存
            self.codes = torch.from_numpy(f["code"][:])  # shape: (N, ...)
            self.labels = torch.tensor(f["label"][:], dtype=torch.long)
            self.paths = [p.decode("utf-8") for p in f["path"][:]]
        self.length = len(self.codes)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.codes[idx], self.labels[idx], self.paths[idx]


class H5DatasetInMemoryWithAug(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            # 一次性全部加载到内存
            self.codes = torch.from_numpy(f["code"][:])  # shape: (N, ...)
            self.labels = torch.tensor(f["label"][:], dtype=torch.long)
            self.paths = [p.decode("utf-8") for p in f["path"][:]]
        
        self.groups = defaultdict(list)
        for idx, p in enumerate(self.paths):
            self.groups[p].append(idx)
        self.unique_paths = list(self.groups.keys())
        self.length = len(self.unique_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        aug_indices = self.groups[self.unique_paths[idx]]
        # 随机选一个增强版本
        chosen_idx = random.choice(aug_indices)
        return (self.codes[chosen_idx], self.labels[chosen_idx], self.paths[chosen_idx])


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable
        self.feature_files = [f"{i}.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


def build_imagenet(args, transform):
    return ImageFolderWithPath(args.data_path, transform=transform)

def merge_h5_datasets(path):
    all_codes = []
    all_labels = []
    all_paths = []
    for f in tqdm(sorted(glob(os.path.join(path, "*.h5")))):
        with h5py.File(f, "r") as f:
            codes = torch.from_numpy(f["code"][:])
            labels = torch.tensor(f["label"][:], dtype=torch.long)
            paths = [p.decode("utf-8") for p in f["path"][:]]

            all_codes.append(codes)
            all_labels.append(labels)
            all_paths.extend(paths)  # paths 是列表，直接扩展

    # 拼接成一个大数组/张量
    all_codes = torch.cat(all_codes, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 保存到新的 h5 文件
    output_file = os.path.join(path, "merged.h5")
    with h5py.File(output_file, "w") as f:
        f.create_dataset("code", data=all_codes.numpy())
        f.create_dataset("label", data=all_labels.numpy())
        # 保存字符串需要转成字节
        path_bytes = [p.encode("utf-8") for p in all_paths]
        f.create_dataset("path", data=np.array(path_bytes, dtype="S"))

def get_code_dataset(path, rank):
    if rank == 0:
        if "merged.h5" not in os.listdir(path):
            print("Merging h5 datasets...")
            merge_h5_datasets(path)
        else:
            print("merged.h5 already exists, skipping merging.")
    dist.barrier()

    assert os.path.exists(os.path.join(path, "merged.h5")), \
        f"merged.h5 not found in {path} after merging."

    return H5DatasetInMemoryWithAug(os.path.join(path, "merged.h5"))

def build_imagenet_code(args):
    code_path = args.code_path
    if not os.path.exists(code_path):
        base_path = os.path.dirname(os.path.abspath(__file__))
        code_path = os.path.join(base_path, code_path)
        assert os.path.exists(code_path), f"code_path {code_path} does not exist"
    return get_code_dataset(code_path, args.rank)
