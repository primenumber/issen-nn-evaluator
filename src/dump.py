import math
import os
import sys
import json

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import PatternBasedV2, BASE_PATTERN_BITS
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

model_path = Path(sys.argv[1])

output_dir_path = Path(sys.argv[2])

output_dir_path.mkdir()

dtype = torch.bfloat16

saved = torch.load(model_path)
mparam = saved["model_param"]
front = mparam["front"]
middle = mparam["middle"]
back = mparam["back"]
model = PatternBasedV2(front, middle, back)
model.load_state_dict(saved["state_dict"])
model.to(device)

config = {
    "patterns": ["{:016x}".format(pattern) for pattern in BASE_PATTERN_BITS],
    "front": front,
    "middle": middle,
    "back": back,
}


json.dump(config, Path(output_dir_path, "config.json").open(mode="w"))

total_params = 0
for param_tensor in model.state_dict():
    tensor = model.state_dict()[param_tensor]
    total_params += torch.numel(tensor)
    tensor_flat = torch.flatten(tensor).to(torch.float32).to("cpu")
    print(param_tensor, tensor.size(), tensor.min(), tensor.max(), tensor_flat)
    with Path(output_dir_path, param_tensor).open("wb") as f:
        tensor_flat.numpy().ravel().tofile(f)

print(total_params)
