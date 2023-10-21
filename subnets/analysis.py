import torch.nn as nn
from .GDN import GDN


class AnalysisNet(nn.Module):
    def __init__(self):
        super(AnalysisNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2
            ),
            GDN(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2
            ),
            GDN(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2
            ),
            GDN(64),
            nn.Conv2d(
                in_channels=64, out_channels=96, kernel_size=5, stride=2, padding=2
            ),
        )

    def forward(self, x):
        return self.model(x)
