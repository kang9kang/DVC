import torch
import torch.nn as nn


class AnalysisPriorNet(nn.Module):
    def __init__(self):
        super(AnalysisPriorNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2
            ),
        )

    def forward(self, x):
        return self.model(torch.abs(x))
