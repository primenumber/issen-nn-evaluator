import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import PatternBasedLarge, PatternBasedSmall

model = torch.load("workdir/reversei_pattern_based.pth")
model.eval()

total_params = 0
for param_tensor in model.state_dict():
    total_params += torch.numel(model.state_dict()[param_tensor])
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("total params:", total_params)
