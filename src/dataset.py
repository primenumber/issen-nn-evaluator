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
            players = []
            opponents = []
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
                players.append(player)
                opponents.append(opponent)
                scores.append(score)
                if len(players) == limit:
                    break
            print()
        self._dtype = dtype
        self._players = np.array(players, dtype=np.uint64)
        self._opponents = np.array(opponents, dtype=np.uint64)
        self._scores = np.array(scores, dtype=np.int32)

    def __len__(self):
        return len(self._scores)

    def __getitem__(self, idx: int):
        player_bits = list(map(int, format(self._players[idx], "064b")))
        opponent_bits = list(map(int, format(self._opponents[idx], "064b")))
        X = torch.zeros([2, 64], dtype=torch.int32)
        y = torch.zeros([1], dtype=self._dtype)
        X[0] = torch.tensor(player_bits, dtype=torch.int32)
        X[1] = torch.tensor(opponent_bits, dtype=torch.int32)
        X = torch.reshape(X, [2, 8, 8])
        if random.random() > 0.5:
            X = torch.transpose(X, 1, 2)
        if random.random() > 0.5:
            X = torch.flip(X, [1])
        if random.random() > 0.5:
            X = torch.flip(X, [2])
        X = torch.reshape(X, [2, 64])
        y[0] = int(self._scores[idx])
        return X, y

