import torch
import torch.nn as nn

from .GDN import GDN


class SynthesisNet(nn.Module):
    def __init__(self):
        super(SynthesisNet, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=96,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            GDN(64, inverse=True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            GDN(64, inverse=True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            GDN(64, inverse=True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
        )

    def forward(self, x):
        return self.model(x)
