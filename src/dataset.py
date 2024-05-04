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
        self._players = np.reshape(np.frombuffer(np.array(players, dtype='<u8').tobytes(), dtype=np.uint8), [len(players), 8])
        self._opponents = np.reshape(np.frombuffer(np.array(opponents, dtype='<u8').tobytes(), dtype=np.uint8), [len(opponents), 8])
        self._scores = np.array(scores, dtype=np.int32)

    def __len__(self):
        return len(self._scores)

    def __getitem__(self, idx: int):
        x = np.stack((np.unpackbits(self._players[idx]), np.unpackbits(self._opponents[idx])))
        X = torch.tensor(x, dtype=torch.uint8)
        X = torch.reshape(X, [-1, 2, 8, 8])
        if random.random() > 0.5:
            X = torch.transpose(X, 2, 3)
        if random.random() > 0.5:
            X = torch.flip(X, [2])
        if random.random() > 0.5:
            X = torch.flip(X, [3])
        X = torch.reshape(X, [-1, 2, 64])
        y = torch.zeros([1], dtype=self._dtype)
        y[0] = int(self._scores[idx])
        return X, y
