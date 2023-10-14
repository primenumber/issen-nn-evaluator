import torch
from torch import nn


def generate_patterns():
    base_patterns = [
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 14],
        [8, 9, 10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29, 30, 31],
        [0, 1, 2, 3, 4, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 8, 9, 10, 16, 17, 24],
        [3, 4, 10, 11, 17, 18, 24, 25, 32],
        [5, 12, 19, 26, 33, 40],
        [6, 13, 20, 27, 34, 41, 48],
        [7, 20, 21, 28, 35, 42, 49, 56],
    ]
    return base_patterns


class EmbedPattern(nn.Module):
    def __init__(self, pattern_size, input_channels, output_channels):
        super(EmbedPattern, self).__init__()
        self.pattern_size = pattern_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.linear_layer = nn.Linear(pattern_size * input_channels, output_channels, bias=False)

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
        self.num_patterns = 10
        self.middle_channels = middle_channels
        self.grouped_channels = self.front_channels * self.num_patterns
        self.frontend_block_0 = EmbedPattern(
            10, self.input_channels, self.front_channels, 
        )
        self.frontend_block_1 = EmbedPattern(
            8, self.input_channels, self.front_channels
        )
        self.frontend_block_2 = EmbedPattern(
            8, self.input_channels, self.front_channels,
        )
        self.frontend_block_3 = EmbedPattern(
            8, self.input_channels, self.front_channels,
        )
        self.frontend_block_4 = EmbedPattern(
            10, self.input_channels, self.front_channels,
        )
        self.frontend_block_5 = EmbedPattern(
            10, self.input_channels, self.front_channels,
        )
        self.frontend_block_6 = EmbedPattern(
            9, self.input_channels, self.front_channels,
        )
        self.frontend_block_7 = EmbedPattern(
            6, self.input_channels, self.front_channels,
        )
        self.frontend_block_8 = EmbedPattern(
            7, self.input_channels, self.front_channels,
        )
        self.frontend_block_9 = EmbedPattern(
            8, self.input_channels, self.front_channels,
        )
        self.middle_block_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.grouped_channels, self.grouped_channels, 1, groups = self.num_patterns),
            nn.ReLU(),
            nn.Conv1d(self.grouped_channels, self.grouped_channels, 1, groups = self.num_patterns),
            nn.ReLU(),
            nn.Conv1d(self.grouped_channels, self.grouped_channels, 1, groups = self.num_patterns),
        )
        self.middle_block_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.grouped_channels, self.grouped_channels, 1, groups = self.num_patterns),
            nn.ReLU(),
            nn.Conv1d(self.grouped_channels, self.grouped_channels, 1, groups = self.num_patterns),
            nn.ReLU(),
            nn.Conv1d(self.grouped_channels, self.grouped_channels, 1, groups = self.num_patterns),
        )
        self.middle_block_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.grouped_channels, self.grouped_channels, 1, groups = self.num_patterns),
            nn.ReLU(),
            nn.Conv1d(self.grouped_channels, self.grouped_channels, 1, groups = self.num_patterns),
            nn.ReLU(),
            nn.Conv1d(self.grouped_channels, self.grouped_channels, 1, groups = self.num_patterns),
        )
        self.middle_block_last = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.grouped_channels, self.num_patterns * self.middle_channels, 1, groups = self.num_patterns),
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
        m0 = self.frontend_block_0(vx[:, :, :, self.patterns[0]])
        m1 = self.frontend_block_1(vx[:, :, :, self.patterns[1]])
        m2 = self.frontend_block_2(vx[:, :, :, self.patterns[2]])
        m3 = self.frontend_block_3(vx[:, :, :, self.patterns[3]])
        m4 = self.frontend_block_4(vx[:, :, :, self.patterns[4]])
        m5 = self.frontend_block_5(vx[:, :, :, self.patterns[5]])
        m6 = self.frontend_block_6(vx[:, :, :, self.patterns[6]])
        m7 = self.frontend_block_7(vx[:, :, :, self.patterns[7]])
        m8 = self.frontend_block_8(vx[:, :, :, self.patterns[8]])
        m9 = self.frontend_block_9(vx[:, :, :, self.patterns[9]])
        m = torch.stack((m0, m1, m2, m3, m4, m5, m6, m7, m8, m9), dim=1)
        m = torch.reshape(m, [-1, self.num_patterns , self.num_symmetry, self.front_channels])
        m = torch.transpose(m, 2, 3)
        m = torch.reshape(m, [-1, self.num_patterns * self.front_channels, self.num_symmetry])
        b0 = m + self.middle_block_1(m)
        b1 = b0 + self.middle_block_2(b0)
        b2 = b1 + self.middle_block_3(b1)
        b3 = self.middle_block_last(b2)
        b4 = torch.reshape(b3, [-1, self.num_patterns, self.middle_channels, self.num_symmetry])
        b4 = torch.reshape(torch.transpose(b4, 1, 2), [-1, self.middle_channels, self.num_patterns * self.num_symmetry])
        b5 = torch.sum(b4, 2)
        y = self.backend_block(b5)
        return y
