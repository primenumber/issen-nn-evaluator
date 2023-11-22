import torch
from torch import nn
import typing


def generate_patterns():
    base_pattern_bits = [
        0x0000_0000_0000_42FF,
        0x0000_0000_0000_FF00,
        0x0000_0000_00FF_0000,
        0x0000_0000_FF00_0000,
        0x0000_0000_0000_1F1F,
        0x0000_0000_0103_070F,
        0x0000_0001_0306_0C18,
        0x0000_0102_0408_1020,
        0x0001_0204_0810_2040,
        0x0102_0408_1020_4080,
        0x0000_0000_0000_3CBD,
        0x0000_0000_0C0E_0703,
        0x0000_0000_0007_0707,
        0x0000_0000_0000_24FF,
        0x0000_0000_0A06_0F03,
        0x0000_0002_0202_1F03,
    ]

    def to_idx(bits: int) -> list[int]:
        result = []
        for i in range(64):
            if (bits >> i) & 1 == 1:
                result.append(i)
        return result

    return [to_idx(bits) for bits in base_pattern_bits]


class EmbedPattern(nn.Module):
    def __init__(self, pattern_size, input_channels, output_channels):
        super(EmbedPattern, self).__init__()
        self.pattern_size = pattern_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.linear_layer = nn.Linear(
            pattern_size * input_channels, output_channels, bias=False
        )

    def forward(self, x):
        x = torch.reshape(x, [-1, 8, self.pattern_size * self.input_channels])
        return self.linear_layer(x)


class PatternBasedV2(nn.Module):
    def __init__(self, front_channels, middle_channels):
        super(PatternBasedV2, self).__init__()
        self.patterns = generate_patterns()
        self.input_channels = 2
        self.front_channels = front_channels
        self.num_symmetry = 8
        self.num_patterns = len(self.patterns)
        self.middle_channels = middle_channels
        self.grouped_channels = self.front_channels * self.num_patterns
        self.frontend_blocks = nn.ModuleList(
            [
                EmbedPattern(len(pattern), self.input_channels, self.front_channels)
                for pattern in self.patterns
            ]
        )
        self.middle_block_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                self.grouped_channels,
                self.grouped_channels,
                1,
                groups=self.num_patterns,
            ),
            nn.ReLU(),
            nn.Conv1d(
                self.grouped_channels,
                self.grouped_channels,
                1,
                groups=self.num_patterns,
            ),
            nn.ReLU(),
            nn.Conv1d(
                self.grouped_channels,
                self.grouped_channels,
                1,
                groups=self.num_patterns,
            ),
        )
        self.middle_block_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                self.grouped_channels,
                self.grouped_channels,
                1,
                groups=self.num_patterns,
            ),
            nn.ReLU(),
            nn.Conv1d(
                self.grouped_channels,
                self.grouped_channels,
                1,
                groups=self.num_patterns,
            ),
            nn.ReLU(),
            nn.Conv1d(
                self.grouped_channels,
                self.grouped_channels,
                1,
                groups=self.num_patterns,
            ),
        )
        self.middle_block_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                self.grouped_channels,
                self.grouped_channels,
                1,
                groups=self.num_patterns,
            ),
            nn.ReLU(),
            nn.Conv1d(
                self.grouped_channels,
                self.grouped_channels,
                1,
                groups=self.num_patterns,
            ),
            nn.ReLU(),
            nn.Conv1d(
                self.grouped_channels,
                self.grouped_channels,
                1,
                groups=self.num_patterns,
            ),
        )
        self.middle_block_last = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                self.grouped_channels,
                self.num_patterns * self.middle_channels,
                1,
                groups=self.num_patterns,
            ),
        )
        self.backend_block = nn.Sequential(
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, middle_channels, bias=False),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, middle_channels, bias=False),
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, 1),
        )

    def forward(self, x):
        x = torch.reshape(x, [-1, 1, 2, 8, 8])
        x0 = x
        x1 = torch.transpose(x, 3, 4)
        x01 = torch.cat((x0, x1), dim=1)
        x23 = torch.flip(x01, [3])
        x03 = torch.cat((x01, x23), dim=1)
        x47 = torch.flip(x03, [4])
        x07 = torch.cat((x03, x47), dim=1)
        vx = torch.reshape(x07, [-1, 8, 2, 64])
        vm = []
        for i, pattern in enumerate(self.patterns):
            vm.append(self.frontend_blocks[i](vx[:, :, :, pattern]))
        m = torch.stack(vm, dim=1)
        m = torch.reshape(
            m, [-1, self.num_patterns, self.num_symmetry, self.front_channels]
        )
        m = torch.transpose(m, 2, 3)
        m = torch.reshape(
            m, [-1, self.num_patterns * self.front_channels, self.num_symmetry]
        )
        b0 = m + self.middle_block_1(m)
        b1 = b0 + self.middle_block_2(b0)
        b2 = b1 + self.middle_block_3(b1)
        b3 = self.middle_block_last(b2)
        b4 = torch.reshape(
            b3, [-1, self.num_patterns, self.middle_channels, self.num_symmetry]
        )
        b4 = torch.reshape(
            torch.transpose(b4, 1, 2),
            [-1, self.middle_channels, self.num_patterns * self.num_symmetry],
        )
        b5 = torch.sum(b4, 2)
        y = self.backend_block(b5)
        return y
