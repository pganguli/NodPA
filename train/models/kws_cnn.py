"""
Small CNN for Keyword Spotting (Google Speech Commands dataset).

KWS_CNN_S: compact depthwise-separable CNN that classifies 1-second mel-
  spectrogram patches into 10 keyword classes.  Reuses ConvBlock from har_cnn.
"""

import torch
import torch.nn as nn

from .har_cnn import ConvBlock


class KWS_CNN_S(nn.Module):
    def __init__(self, n_channels=1):
        super(KWS_CNN_S, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=n_channels,
            out_channels=28,
            kernel_size=(10, 4),
            padding=0,
            stride1=1,
            stride2=(2, 1),
        )
        self.ip1 = nn.Linear(2 * 2 * 28, 16)
        self.ip2 = nn.Linear(16, 128)
        self.ip3 = nn.Linear(128, 12)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), 2 * 2 * 28)
        x = self.ip1(x)
        x = self.ip2(x)
        x = self.relu(x)
        x = self.ip3(x)
        return x
