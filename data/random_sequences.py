# heartgpt/data/random_sequences.py

import torch
from torch.utils.data import Dataset


class RandomSequenceDataset(Dataset):
    """
    Placeholder dataset that generates completely random token sequences in [0, vocab_size).
    Samples windows of length block_size+1 for next-token modeling.
    """
    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        base_sequence_count: int = 200,
        base_sequence_length: int = 1000,
        seed: int = 48,
    ):
        torch.manual_seed(seed)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.seqs = torch.randint(0, vocab_size, (base_sequence_count, base_sequence_length), dtype=torch.long)
        self.num_seqs, self.seq_len = self.seqs.shape
        if self.seq_len < block_size + 1:
            raise ValueError("Base sequence length must be >= block_size + 1")

    def __len__(self):
        return 10**9  # effectively infinite for random sampling

    def __getitem__(self, idx_unused):
        i = torch.randint(0, self.num_seqs, (1,)).item()
        max_start = self.seq_len - (self.block_size + 1)
        start = torch.randint(0, max_start + 1, (1,)).item()
        seq = self.seqs[i, start : start + self.block_size + 1]
        x = seq[: self.block_size]
        y = seq[1 :]
        return x, y
