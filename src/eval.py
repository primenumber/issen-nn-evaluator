import math
import os
import sys

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import PatternBasedV2
from dataset import ReversiDataset
from lr_schedule import cosine_with_warmup

use_ipex = "USE_IPEX" in os.environ
if use_ipex:
    import intel_extension_for_pytorch as ipex

if use_ipex:
    device = "xpu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device} device")

dtype = torch.bfloat16

saved = torch.load(sys.argv[1])
mparam = saved["model_param"]
front = mparam["front"]
middle = mparam["middle"]
back = mparam["back"]
model = PatternBasedV2(front, middle, back)
model.load_state_dict(saved["state_dict"])
model.to(device)

def eval_sample():
    board = { "player": 18324862011005731840, "opponent": 13514196661811216 }
    player_bits = list(map(int, format(board["player"], "064b")))
    opponent_bits = list(map(int, format(board["opponent"], "064b")))
    X = torch.tensor([[player_bits, opponent_bits]], dtype=torch.int8).to(device)
    pred = model(X)
    print(pred.item())
    exit()

def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    diff_by_stones = [[] for _ in range(64)]
    diffsq_by_stones = [[] for _ in range(64)]
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            diff = torch.flatten(torch.abs(y - pred)).tolist()
            stones = torch.sum(torch.reshape(X, [-1, 128]), dim=1).tolist()
            for (d, s) in zip(diff, stones):
                dsq = d * d
                diffsq_by_stones[s].append(dsq)
                diff_by_stones[s].append(d)

    for (i, (dsq, dd)) in enumerate(zip(diffsq_by_stones, diff_by_stones)):
        if len(dd) == 0:
            continue
        sq_diff = math.sqrt(sum(dsq) / len(dsq))
        avg_diff = sum(dd) / len(dd)
        print(sq_diff, avg_diff)
                

eval_sample()

test_data_file = "workdir/dataset_221009_test.txt"
stones_filter = {i for i in range(14, 60)}
test_data = ReversiDataset(test_data_file, dtype, stones_filter, 33554432)
batch_size = 4096
test_dataloader = DataLoader(
    test_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
)
test_loop(test_dataloader, model)
