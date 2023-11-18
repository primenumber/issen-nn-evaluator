import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import PatternBasedV2

use_ipex = "USE_IPEX" in os.environ
if use_ipex:
    import intel_extension_for_pytorch as ipex

model = torch.load("workdir/reversei_pattern_based.pth")
model.eval()

total_params = 0
for param_tensor in model.state_dict():
    tensor = model.state_dict()[param_tensor]
    total_params += torch.numel(tensor)
    print(param_tensor, tensor.size(), tensor.min(), tensor.max())
    print(" ".join(map(str, torch.flatten(tensor).tolist())))

print("total params:", total_params)
