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


class PatternBlockSmall(nn.Module):
    def __init__(self, pattern_size, input_channels, output_channels):
        super(PatternBlockSmall, self).__init__()
        self.pattern_size = pattern_size
        self.input_channels = input_channels
        middle_channels = 32
        resblock_channels = 32
        self.linear_relu_stack = nn.Sequential(
            nn.Conv1d(pattern_size * input_channels, resblock_channels, 1, bias=False),
            nn.BatchNorm1d(resblock_channels),
            nn.ReLU(),
            nn.Conv1d(resblock_channels, middle_channels, 1, bias=False),
        )
        self.res_block_1 = nn.Sequential(
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(),
            nn.Conv1d(middle_channels, resblock_channels, 1, bias=False),
            nn.BatchNorm1d(resblock_channels),
            nn.ReLU(),
            nn.Conv1d(resblock_channels, resblock_channels, 1, bias=False),
            nn.BatchNorm1d(resblock_channels),
            nn.ReLU(),
            nn.Conv1d(resblock_channels, middle_channels, 1, bias=False),
        )
        self.res_block_2 = nn.Sequential(
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(),
            nn.Conv1d(middle_channels, resblock_channels, 1, bias=False),
            nn.BatchNorm1d(resblock_channels),
            nn.ReLU(),
            nn.Conv1d(resblock_channels, resblock_channels, 1, bias=False),
            nn.BatchNorm1d(resblock_channels),
            nn.ReLU(),
            nn.Conv1d(resblock_channels, middle_channels, 1, bias=False),
        )
        self.last_block = nn.Sequential(
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(),
            nn.Conv1d(middle_channels, output_channels, 1, bias=False),
        )

    def forward(self, x):
        x = torch.reshape(x, [-1, 8, self.pattern_size * self.input_channels])
        x = torch.transpose(x, 1, 2)
        y0 = self.linear_relu_stack(x)
        y1 = self.res_block_1(y0) + y0
        y2 = self.res_block_2(y1) + y1
        return self.last_block(y2)


class PatternBlockLarge(nn.Module):
    def __init__(self, pattern_size, input_channels, output_channels):
        super(PatternBlockLarge, self).__init__()
        self.pattern_size = pattern_size
        self.input_channels = input_channels
        middle_channels = 64
        resblock_channels = 128
        self.linear_relu_stack = nn.Sequential(
            nn.Conv1d(pattern_size * input_channels, resblock_channels, 1, bias=False),
            nn.BatchNorm1d(resblock_channels),
            nn.ReLU(),
            nn.Conv1d(resblock_channels, middle_channels, 1, bias=False),
        )
        self.res_block_1 = nn.Sequential(
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(),
            nn.Conv1d(middle_channels, resblock_channels, 1, bias=False),
            nn.BatchNorm1d(resblock_channels),
            nn.ReLU(),
            nn.Conv1d(resblock_channels, resblock_channels, 1, bias=False),
            nn.BatchNorm1d(resblock_channels),
            nn.ReLU(),
            nn.Conv1d(resblock_channels, middle_channels, 1, bias=False),
        )
        self.res_block_2 = nn.Sequential(
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(),
            nn.Conv1d(middle_channels, resblock_channels, 1, bias=False),
            nn.BatchNorm1d(resblock_channels),
            nn.ReLU(),
            nn.Conv1d(resblock_channels, resblock_channels, 1, bias=False),
            nn.BatchNorm1d(resblock_channels),
            nn.ReLU(),
            nn.Conv1d(resblock_channels, middle_channels, 1, bias=False),
        )
        self.last_block = nn.Sequential(
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(),
            nn.Conv1d(middle_channels, output_channels, 1, bias=False),
        )

    def forward(self, x):
        x = torch.reshape(x, [-1, 8, self.pattern_size * self.input_channels])
        x = torch.transpose(x, 1, 2)
        y0 = self.linear_relu_stack(x)
        y1 = self.res_block_1(y0) + y0
        y2 = self.res_block_2(y1) + y1
        return self.last_block(y2)


class PatternBasedSmall(nn.Module):
    def __init__(self, middle_channel_expansion):
        super(PatternBasedSmall, self).__init__()
        self.flatten = nn.Flatten()
        self.patterns = generate_patterns()
        input_channels = 2
        self.frontend_block_0 = PatternBlockSmall(
            10, input_channels, middle_channel_expansion
        )
        self.frontend_block_1 = PatternBlockSmall(
            8, input_channels, middle_channel_expansion
        )
        self.frontend_block_2 = PatternBlockSmall(
            8, input_channels, middle_channel_expansion
        )
        self.frontend_block_3 = PatternBlockSmall(
            8, input_channels, middle_channel_expansion
        )
        self.frontend_block_4 = PatternBlockSmall(
            10, input_channels, middle_channel_expansion
        )
        self.frontend_block_5 = PatternBlockSmall(
            10, input_channels, middle_channel_expansion
        )
        self.frontend_block_6 = PatternBlockSmall(
            9, input_channels, middle_channel_expansion
        )
        self.frontend_block_7 = PatternBlockSmall(
            6, input_channels, middle_channel_expansion
        )
        self.frontend_block_8 = PatternBlockSmall(
            7, input_channels, middle_channel_expansion
        )
        self.frontend_block_9 = PatternBlockSmall(
            8, input_channels, middle_channel_expansion
        )
        self.backend_block = nn.Sequential(
            nn.BatchNorm1d(middle_channel_expansion),
            nn.ReLU(),
            nn.Linear(middle_channel_expansion, middle_channel_expansion, bias=False),
            nn.BatchNorm1d(middle_channel_expansion),
            nn.ReLU(),
            nn.Linear(middle_channel_expansion, middle_channel_expansion, bias=False),
            nn.BatchNorm1d(middle_channel_expansion),
            nn.ReLU(),
            nn.Linear(middle_channel_expansion, 1),
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
        m = torch.cat((m0, m1, m2, m3, m4, m5, m6, m7, m8, m9), dim=2)
        m = torch.transpose(m, 1, 2)
        m = torch.sum(m, 1)
        y = self.backend_block(m)
        return y


class PatternBasedLarge(nn.Module):
    def __init__(self, middle_channel_expansion):
        super(PatternBasedLarge, self).__init__()
        self.flatten = nn.Flatten()
        self.patterns = generate_patterns()
        input_channels = 2
        self.frontend_block_0 = PatternBlockLarge(
            10, input_channels, middle_channel_expansion
        )
        self.frontend_block_1 = PatternBlockLarge(
            8, input_channels, middle_channel_expansion
        )
        self.frontend_block_2 = PatternBlockLarge(
            8, input_channels, middle_channel_expansion
        )
        self.frontend_block_3 = PatternBlockLarge(
            8, input_channels, middle_channel_expansion
        )
        self.frontend_block_4 = PatternBlockLarge(
            10, input_channels, middle_channel_expansion
        )
        self.frontend_block_5 = PatternBlockLarge(
            10, input_channels, middle_channel_expansion
        )
        self.frontend_block_6 = PatternBlockLarge(
            9, input_channels, middle_channel_expansion
        )
        self.frontend_block_7 = PatternBlockLarge(
            6, input_channels, middle_channel_expansion
        )
        self.frontend_block_8 = PatternBlockLarge(
            7, input_channels, middle_channel_expansion
        )
        self.frontend_block_9 = PatternBlockLarge(
            8, input_channels, middle_channel_expansion
        )
        self.backend_block = nn.Sequential(
            nn.BatchNorm1d(middle_channel_expansion),
            nn.ReLU(),
            nn.Linear(middle_channel_expansion, middle_channel_expansion, bias=False),
            nn.BatchNorm1d(middle_channel_expansion),
            nn.ReLU(),
            nn.Linear(middle_channel_expansion, middle_channel_expansion, bias=False),
            nn.BatchNorm1d(middle_channel_expansion),
            nn.ReLU(),
            nn.Linear(middle_channel_expansion, middle_channel_expansion, bias=False),
            nn.BatchNorm1d(middle_channel_expansion),
            nn.ReLU(),
            nn.Linear(middle_channel_expansion, 1),
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
        m = torch.cat((m0, m1, m2, m3, m4, m5, m6, m7, m8, m9), dim=2)
        m = torch.transpose(m, 1, 2)
        m = torch.sum(m, 1)
        y = self.backend_block(m)
        return y
