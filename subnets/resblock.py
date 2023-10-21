import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
        )
        self.adapt_conv = (
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=1,
            )
            if input_channel != output_channel
            else None
        )

    def forward(self, x):
        out = self.model(x)
        if self.adapt_conv is not None:
            x = self.adapt_conv(x)
        return out + x
