# heartgpt/data/labeled_sequence.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import os

from tokenise.preprocess import tokenise_biosignal

class LabeledSequenceDataset(Dataset):
    """
    Dataset for sequence-level labels or token-wise labels.

    Primary mode: load from a CSV with columns:
        - context_path: path to CSV of raw signal (no header)
        - label_path: path to CSV of binary labels (0/1) per time step

    Both context and label sequences are tokenized / loaded and padded/truncated to block_size.

    Fallback mode (use_random_fallback=True or if file missing/invalid): produces
        x: random ints in [0, vocab_size)
        y: random binary sequence (0/1), both of shape (block_size,)

    Returns:
        x: torch.LongTensor of shape (block_size,)
        y: torch.FloatTensor of shape (block_size,) with values 0.0 or 1.0
    """

    def __init__(
        self,
        labels_csv: str = None,
        block_size: int = 500,
        vocab_size: int = 101,
        use_random_fallback: bool = False,
        seed: int = 42,
        pad_token: int = 0,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.use_random_fallback = use_random_fallback

        self.rows = []
        if labels_csv and os.path.isfile(labels_csv) and not use_random_fallback:
            try:
                df = pd.read_csv(labels_csv)
                if "context_path" not in df.columns or "label_path" not in df.columns:
                    raise ValueError("Label CSV must have 'context_path' and 'label_path' columns.")
                self.rows = df.to_dict(orient="records")
            except Exception:
                # fallback to random if label file is malformed
                self.use_random_fallback = True
                self.rows = []
        else:
            # no valid label CSV; will use random fallback
            self.use_random_fallback = True
            self.rows = []

    def __len__(self):
        if self.use_random_fallback:
            # large random pool
            return 10000
        return len(self.rows)

    def _load_and_tokenize_context(self, context_path):
        arr = pd.read_csv(context_path, header=None).values
        tokenized = tokenise_biosignal(arr, max_length=self.block_size)  # numpy array
        # collapse channel if needed
        if tokenized.ndim == 2 and tokenized.shape[0] == 1:
            seq = tokenized[0]
        elif tokenized.ndim == 1:
            seq = tokenized
        else:
            seq = tokenized[0]
        # pad or truncate
        if len(seq) < self.block_size:
            pad = np.full((self.block_size - len(seq),), self.pad_token, dtype=int)
            seq = np.concatenate([pad, seq])
        elif len(seq) > self.block_size:
            seq = seq[-self.block_size :]
        # ensure within vocab
        seq = np.clip(seq, 0, self.vocab_size - 1)
        return torch.tensor(seq, dtype=torch.long)

    def _load_label_sequence(self, label_path):
        arr = pd.read_csv(label_path, header=None).values.flatten()
        # ensure binary
        arr = (arr > 0).astype(int)
        if len(arr) < self.block_size:
            pad = np.zeros((self.block_size - len(arr),), dtype=int)
            arr = np.concatenate([pad, arr])
        elif len(arr) > self.block_size:
            arr = arr[-self.block_size :]
        return torch.tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx):
        if self.use_random_fallback:
            # random x and y
            x = torch.randint(0, self.vocab_size, (self.block_size,), dtype=torch.long)
            y = torch.randint(0, 2, (self.block_size,), dtype=torch.float32)
            return x, y

        row = self.rows[idx]
        context_path = row["context_path"]
        label_path = row["label_path"]

        x = self._load_and_tokenize_context(context_path)

        try:
            y = self._load_label_sequence(label_path)
        except Exception:
            # if label loading fails, fallback to all-zero
            y = torch.zeros((self.block_size,), dtype=torch.float32)

        return x, y
