import torch.nn as nn


class SynthesisMvNet(nn.Module):
    def __init__(self):
        super(SynthesisMvNet, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(
                in_channels=128,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        return self.model(x)
