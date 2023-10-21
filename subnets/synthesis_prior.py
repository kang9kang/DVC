import torch
import torch.nn as nn


class SynthesisPriorNet(nn.Module):
    def __init__(self):
        super(SynthesisPriorNet, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        return torch.exp(self.model(x))
