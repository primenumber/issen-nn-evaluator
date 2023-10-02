import time
import random
from typing import Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import PatternBasedLarge, PatternBasedSmall


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class ReversiDataset(Dataset):
    def __init__(
        self, dataset_file: str, stone_filter: set[int], limit: Optional[int] = None
    ):
        with open(dataset_file) as f:
            n = int(f.readline())
            if limit is not None:
                n = min(n, limit)
            players = []
            opponents = []
            scores = []
            for i in range(n):
                if i % 131072 == 0:
                    print(f"Loading... {i}")
                player_s, opponent_s, score, _ = f.readline().split()
                player = int(player_s, base=16)
                opponent = int(opponent_s, base=16)
                stone_count = bin(player | opponent).count("1")
                if stone_count not in stone_filter:
                    continue
                players.append(player)
                opponents.append(opponent)
                scores.append(score)
        self._players = players
        self._opponents = opponents
        self._scores = scores

    def __len__(self):
        return len(self._scores)

    def __getitem__(self, idx: int):
        global device
        player_bits = list(map(int, format(self._players[idx], "064b")))
        opponent_bits = list(map(int, format(self._opponents[idx], "064b")))
        X = torch.zeros([2, 64])
        y = torch.zeros([1])
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


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    start = time.perf_counter()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    end = time.perf_counter()
    print(f"Time per Epoch: {end - start:>5f}s")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f}")


# model = PatternBasedLarge(32).to(device)
model = PatternBasedSmall(8).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.03, end_factor=1.0, total_iters=5
)
loss_fn = nn.MSELoss()

# batch_size = 128  # for PatternBasedLarge
batch_size = 64  # for PatternBasedSmall
epochs = 20

train_data_file = "workdir/dataset_221009_train.txt"
test_data_file = "workdir/dataset_221009_test.txt"

# stones_filter = {i for i in range(50, 55)}
# train_data = ReversiDataset(train_data_file, stones_filter)
# test_data = ReversiDataset(test_data_file, stones_filter)
stones_filter = {i for i in range(14, 60)}
train_data = ReversiDataset(train_data_file, stones_filter, 16777216)
test_data = ReversiDataset(test_data_file, stones_filter, 16777216)

train_dataloader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=8
)
test_dataloader = DataLoader(
    test_data, batch_size=batch_size, shuffle=True, num_workers=8
)

for t in range(epochs):
    print(f"[[[ Epoch {t+1} ]]]\n--------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    scheduler.step()
print("Save model...")
torch.save(model, "workdir/reversei_pattern_based.pth")
print("Done!")
