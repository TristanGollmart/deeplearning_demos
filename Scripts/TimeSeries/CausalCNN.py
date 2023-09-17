import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class causalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, use_latest_obs=False, **kwargs):
        super(causalConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert kernel_size % 2 > 0, "kernel_size must be of uneven dimension"
        self.use_latest_obs = use_latest_obs
        self.padding = dilation * (kernel_size - 1) + 1 - use_latest_obs
        self.dilation = dilation

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=0, dilation=dilation, **kwargs)

    def forward(self, input):
        # expects input shape [B, C, T] ,
        # returns shape [B, C, T]
        x = F.pad(input, (int(self.padding), 0))
        out = self.conv1(x)
        if not self.use_latest_obs:
            out = out[:, :, :-1] #last element uses x(t) and has to be discarded to respect causality
        return out


class CausalCNN(nn.Module):
    def __init__(self, in_channels: list, out_channels: list, kernel_sizes: list,
                 strides=[], dilations=[]):
        super(CausalCNN, self).__init__()

        assert len(in_channels) == len(out_channels), "length of parameter lists must match"

        if len(strides) > 0:
            self.strides = strides
        else:
            self.strides = np.ones(len(in_channels), dtype=int)
        if len(dilations) > 0:
            self.dilations = dilations
        else:
            self.dilations = np.ones(len(in_channels), dtype=int)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        layers = []
        for i in range(len(in_channels)):
            layers.append(causalConv1D(self.in_channels[i],
                                       self.out_channels[i],
                                       kernel_sizes[i],
                                       dilation=self.dilations[i],
                                       stride=self.strides[i],
                                       use_latest_obs=(i>0)))

        self.cnn_block = nn.ModuleList(layers)
        self.loss = nn.MSELoss()


    def forward(self, input):
        assert self.in_channels[0] == np.shape(input)[1], "input channels does not match expected number of channels"
        x = input
        for l in self.cnn_block:
            x0 = x
            x = self.activation(l(x))
            # residual architecture: later cnn layers can add to previous outputs
            # x = x0 + x
        return x

    def get_loss(self, y, y_pred):
        # nfirst: n first elements to discard since they are mixing 0 values from padding
        nfirst = 0
        for i, kernel_size in enumerate(self.kernel_sizes):
           nfirst += (kernel_size-1) * self.dilations[i]

        return self.loss(y[:, :, nfirst:], y_pred[:, :, nfirst:])


# Examples
c = causalConv1D(in_channels=1, out_channels=1, kernel_size=3, stride=1, dilation=1)
input = torch.ones((1, 1, 100))
output = c(input)
print(output)

input[0, 0, 9] = 100
output2 = c(input)

# normal conv testing: central kernel or right skewed?
normal_conv = nn.Conv1d(1, 1, 3, 1, padding="same", dilation=2)
output_normal_conv = normal_conv(input)
#
print(output2)

cmodel = CausalCNN(in_channels=[1, 5, 3], out_channels=[5, 3, 1], kernel_sizes=[3, 3, 3], dilations=[1, 3, 5])
cmodel.to(device)
output = cmodel(input)
print(output)


# Fitting
def generate_data(shape: tuple):
    B, C, T = shape
    data = np.zeros(shape)
    ampl = np.random.rand(B) + 1
    phase = (np.random.rand(B) + 1) * np.pi
    freq = (np.random.rand(B) + 1) * np.pi
    for t in range(T):
        data[:, :, t] = ampl * np.sin(t*freq + phase)
    return data

input = generate_data(shape=(1,1,100))

nepochs = 100
optimizer = torch.optim.AdamW(cmodel.parameters(), lr=0.01)
for i in range(nepochs):
    y_pred = cmodel(input)
    loss = cmodel.get_loss(input, y_pred)
    print(f"step {i}: loss: {loss:.4f}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


import matplotlib.pyplot as plt

plt.plot(y_pred)
plt.plot(input)
plt.show()

print("finished")
