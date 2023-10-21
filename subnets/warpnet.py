import torch
import torch.nn as nn

from .resblock import ResBlock


class WarpNet(nn.Module):
    def __init__(self):
        super(WarpNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList(
            [ResBlock(64, 64, 3) for _ in range(6)]
            + [
                nn.Conv2d(
                    in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1
                )
            ]
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        feature_ext = self.feature_extractor(x)
        c0 = self.convs[0](feature_ext)
        c0_p = self.avgpool(c0)
        c1 = self.convs[1](c0_p)
        c1_p = self.avgpool(c1)
        c2 = self.convs[2](c1_p)
        c3 = self.convs[3](c2)
        c3_u = self.upsample(c3) + c1
        c4 = self.convs[4](c3_u)
        c4_u = self.upsample(c4) + c0
        c5 = self.convs[5](c4_u)
        res = self.convs[6](c5)
        return res
