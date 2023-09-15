import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class causalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(causalConv1D, self).__init__()
        self. in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert kernel_size % 2 > 0, "kernel_size must be of uneven dimension"
        self.padding = dilation * (kernel_size - 1) / 2 + 1
        self.dilation = dilation

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=0, dilation=dilation)

    def forward(self, input):
        x = F.pad(input, (int(self.padding), -int(self.padding)))
        x = self.conv1(x)
        return x


c = causalConv1D(in_channels=1, out_channels=1, kernel_size=3, stride=1, dilation=1)
input = torch.ones((1, 1, 100))
output = c(input)
print(output)

input[0, 0, 9] = 100
output2 = c(input)
print(output2)