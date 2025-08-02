# heartgpt/data/ecg_dataset.py

# I do not own the data, and therefore I cannot upload it to github. 
# please refer to https://arxiv.org/abs/2407.20775 for instructions on the ecg data used

import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np


class ECGDataset(Dataset):
    """
    Loads .mat file expecting key 'ecg_store' with shape (num_traces, trace_length).
    Provides random cropped windows of length block_size+1 for next-token prediction.

    """
    def __init__(self, mat_path, block_size, split="train", train_frac=0.9, seed=42):
        torch.manual_seed(seed)
        data_mat = scipy.io.loadmat(mat_path)
        if "ecg_store" not in data_mat:
            raise KeyError(f"'ecg_store' not found in {mat_path}")
        data = data_mat["ecg_store"]  # (N, L)
        perm = np.random.permutation(data.shape[0])
        data = data[perm, :]
        split_idx = int(train_frac * data.shape[0])
        if split == "train":
            subset = data[:split_idx]
        elif split in ("val", "validation"):
            subset = data[split_idx:]
        else:
            raise ValueError(f"Unknown split {split}")
        self.block_size = block_size
        self.tensor = torch.tensor(subset, dtype=torch.long)  # (N_split, L)
        self.num_traces, self.trace_len = self.tensor.shape
        if self.trace_len < block_size + 1:
            raise ValueError(f"Trace length {self.trace_len} is less than block_size+1 ({block_size+1})")

    def __len__(self):
        return 10**9  # effectively infinite for random sampling

    def __getitem__(self, idx_unused):
        i = torch.randint(0, self.num_traces, (1,)).item()
        max_start = self.trace_len - (self.block_size + 1)
        start = torch.randint(0, max_start + 1, (1,)).item()
        seq = self.tensor[i, start : start + self.block_size + 1]
        x = seq[: self.block_size]
        y = seq[1 :]
        return x, y
