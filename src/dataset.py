import torch
from torch.utils.data import Dataset
from typing import Optional
import numpy as np
import random


class ReversiDataset(Dataset):
    def __init__(
            self, dataset_file: str, dtype: torch.dtype, stone_filter: set[int], limit: Optional[int] = None
    ):
        with open(dataset_file) as f:
            n = int(f.readline())
            boards = []
            scores = []
            for i in range(n):
                if i % 65536 == 0:
                    print(f"Loading from {dataset_file} ... {i}/{n}\r", end="")
                player_s, opponent_s, score, _ = f.readline().split()
                player = int(player_s, base=16)
                opponent = int(opponent_s, base=16)
                stone_count = bin(player | opponent).count("1")
                if stone_count not in stone_filter:
                    continue
                boards.append([player, opponent])
                scores.append(score)
                if len(boards) == limit:
                    break
            print()
        self._dtype = dtype
        self._boards = np.reshape(np.frombuffer(np.array(boards, dtype='<u8').tobytes(), dtype=np.uint8), [len(boards), 16])
        self._scores = np.array(scores, dtype=np.int32)

    def __len__(self):
        return len(self._scores)

    def __getitem__(self, idx: int):
        X = torch.tensor(np.unpackbits(self._boards[idx]), dtype=torch.uint8)
        y = torch.zeros([1], dtype=self._dtype)
        y[0] = int(self._scores[idx])
        return X, y


def data_augumentation(t: torch.Tensor) -> torch.Tensor:
    X = torch.reshape(t, [-1, 2, 8, 8])
    if random.random() > 0.5:
        X = torch.transpose(X, 2, 3)
    if random.random() > 0.5:
        X = torch.flip(X, [2])
    if random.random() > 0.5:
        X = torch.flip(X, [3])
    return torch.reshape(X, [-1, 2, 64])
