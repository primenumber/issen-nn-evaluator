import os
import time
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import PatternBasedV2
from dataset import ReversiDataset

use_ipex = "USE_IPEX" in os.environ
if use_ipex:
    import intel_extension_for_pytorch as ipex

    print(ipex.xpu.get_device_name(0))

if use_ipex:
    device = "xpu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device} device")

dtype = torch.bfloat16

writer = SummaryWriter()

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    start = time.perf_counter()
    for batch, (X, y) in enumerate(dataloader):
        if batch == 100:
            end = time.perf_counter()
            current = batch * len(X)
            print(f"Time per 100 batch: {end - start:>5f}s, [{current:>5d}/{size:>5d}]")

        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch * num_batches + batch)
        #    loss, current = loss.item(), batch * len(X)
        #    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    end = time.perf_counter()
    print(f"Time per Epoch: {end - start:>5f}s")


def test_loop(dataloader, model, loss_fn, epoch):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f}")
    writer.add_scalar('Loss/test', test_loss, epoch + 1)


front = 256
middle = 256
back = 32
# model = PatternBasedV2(32, 32, 32).to(device)
model = PatternBasedV2(front, middle, back).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

if use_ipex:
    model, optimizer = ipex.optimize(model, dtype=dtype, optimizer=optimizer)
elif device == "cuda":
    model = torch.compile(model)

scheduler1 = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.2, end_factor=1.0, total_iters=5
)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma = 0.9
)
loss_fn = nn.MSELoss()

batch_size = 4096  # for PatternBasedLarge
# batch_size = 4096  # for PatternBasedSmall
epochs = 60

train_data_file = "workdir/dataset_221009_train.txt"
test_data_file = "workdir/dataset_221009_test.txt"

# stones_filter = {i for i in range(50, 55)}
stones_filter = {i for i in range(14, 60)}
# train_data = ReversiDataset(train_data_file, stones_filter, 100000)
# test_data = ReversiDataset(test_data_file, stones_filter, 100000)
# train_data = ReversiDataset(train_data_file, stones_filter, 1048576)
# test_data = ReversiDataset(test_data_file, stones_filter, 1048576)
train_data = ReversiDataset(train_data_file, dtype, stones_filter, -1)
test_data = ReversiDataset(test_data_file, dtype, stones_filter, 33554432)

train_dataloader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
)
test_dataloader = DataLoader(
    test_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
)

for t in range(epochs):
    print(f"Epoch {t+1}")
    train_loop(train_dataloader, model, loss_fn, optimizer, t)
    test_loop(test_dataloader, model, loss_fn, t)
    scheduler1.step()
    scheduler2.step()
    print("Save tmp model...")
    torch.save(model, f"workdir/nnue_{front}_{middle}_{back}_e{t}.pth")
print("Save model...")
torch.save(model, f"workdir/nnue_{front}_{middle}_{back}.pth")
print("Done!")
