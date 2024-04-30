import torch
from torch import nn
import typing


BASE_PATTERN_BITS = [
    0x0000_0000_0000_40FF,
    0x0000_0000_0000_FF00,
    0x0000_0000_00FF_0000,
    0x0000_0000_FF00_0000,
    0x0000_0000_0000_0F1F,
    0x0000_0000_0003_070F,
    0x0000_0001_0306_0C18,
    0x0000_0102_0408_1020,
    0x0001_0204_0810_2040,
    0x0102_0408_1020_4080,
    0x0000_0000_0000_1CBD,
    0x0000_0000_080E_0703,
    0x0000_0000_0007_0707,
    0x0000_0000_0000_04FF,
    0x0000_0000_0806_0F03,
    0x0000_0000_0202_1F03,
]

def generate_patterns() -> list[list[int]]:
    def to_idx(bits: int) -> list[int]:
        result = []
        for i in range(64):
            if (bits >> i) & 1 == 1:
                result.append(i)
        return result

    return [to_idx(bits) for bits in BASE_PATTERN_BITS]


def generate_pattern_indexer(patterns: list[list[int]]) -> tuple[list[list[int]], list[int], int]:
    def to_indexer(pattern: list[int]) -> list[list[int]]:
        result = [0 for _ in range(64)]
        for i, pos in enumerate(pattern):
            result[pos] = 3 ** i
        return result

    offset = 0
    mat = []
    bias = []
    for pattern in patterns:
        mat.append(to_indexer(pattern))
        bias.append(offset)
        offset += 3 ** len(pattern)

    return (mat, bias, offset)


class PatternBasedV2(nn.Module):
    def __init__(self, front_channels, middle_channels, back_channels):
        super(PatternBasedV2, self).__init__()
        self.patterns = generate_patterns()
        idx_mat, idx_bias, total_idx = generate_pattern_indexer(self.patterns)
        self.indexer_mat = torch.transpose(torch.tensor(idx_mat, dtype=torch.int32), 0, 1).to(torch.float32)
        self.indexer_bias = torch.reshape(torch.tensor(idx_bias, dtype=torch.int32), [1, len(self.patterns)])
        self.input_channels = 2
        self.front_channels = front_channels
        self.num_symmetry = 8
        self.num_patterns = len(self.patterns)
        self.back_channels = back_channels
        self.embedding = nn.EmbeddingBag(total_idx, front_channels, max_norm = 1.0, mode="sum")
        self.backend_block = nn.Sequential(
            nn.Linear(front_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, back_channels),
            nn.ReLU(),
            nn.Linear(back_channels, 1),
        )

    def forward(self, x):
        x = torch.reshape(x.to(torch.float32), [-1, 1, 2, 8, 8])
        xp = x[:, :, 0, :, :]
        xo = x[:, :, 1, :, :]
        x = xp + 2.0 * xo
        x0 = x
        x1 = torch.transpose(x, 2, 3)
        x01 = torch.cat((x0, x1), dim=1)
        x23 = torch.flip(x01, [2])
        x03 = torch.cat((x01, x23), dim=1)
        x47 = torch.flip(x03, [3])
        x07 = torch.cat((x03, x47), dim=1)
        vx = torch.reshape(x07, [-1, 64])
        s = torch.matmul(vx, self.indexer_mat.to(x.device)) + self.indexer_bias.to(x.device)
        s = torch.reshape(s, [-1, self.num_symmetry * len(self.patterns)]).to(torch.int32)
        m = self.embedding(s)
        y = self.backend_block(m)
        return y
