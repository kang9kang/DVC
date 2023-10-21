from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .flowwarp import FlowWarpper


class MEBasicBlock(nn.Module):
    def __init__(self, layer):
        super(MEBasicBlock, self).__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(8, 32, 7, 1, 3),
                nn.Conv2d(32, 64, 7, 1, 3),
                nn.Conv2d(64, 32, 7, 1, 3),
                nn.Conv2d(32, 16, 7, 1, 3),
                nn.Conv2d(16, 2, 7, 1, 3),
            ]
        )

        self.relu = nn.ReLU()
        self.init_conv_params(layer)

    def init_conv_params(self, layer):
        params_path = "flow_pretrain_np/modelL"

        for i, conv in enumerate(self.convs):
            conv.weight.data = torch.from_numpy(
                np.load(params_path + str(layer) + "_F-" + str(i + 1) + "-weight.npy")
            )
            conv.bias.data = torch.from_numpy(
                np.load(params_path + str(layer) + "_F-" + str(i + 1) + "-bias.npy")
            )

    def forward(self, x):
        for i in range(5):
            x = self.convs[i](x)
            if i != 4:
                x = self.relu(x)
        return x


class MESpynet(nn.Module):
    def __init__(self):
        super(MESpynet, self).__init__()
        self.blocklist = nn.ModuleList([MEBasicBlock(i + 1) for i in range(4)])
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.flowwarp = FlowWarpper()

    def forward(self, img_input, img_ref):
        batch_size = img_input.size(0)

        img_input_list = [img_input]
        img_ref_list = [img_ref]
        for i in range(3):
            img_input_list.append(
                F.avg_pool2d(img_input_list[-1], kernel_size=2, stride=2)
            )
            img_ref_list.append(F.avg_pool2d(img_ref_list[-1], kernel_size=2, stride=2))

        shape = img_input_list[-1].size()
        zeroshape = (batch_size, 2, shape[2] // 2, shape[3] // 2)

        flowfileds = torch.zeros(zeroshape).cuda()

        for i, block in enumerate(self.blocklist):
            flowfiledsupsample = self.upsample(flowfileds) * 2
            flowfileds = flowfiledsupsample + block(
                torch.cat(
                    (
                        img_input_list[3 - i],
                        self.flowwarp(img_ref_list[3 - i], flowfiledsupsample),
                        flowfiledsupsample,
                    ),
                    dim=1,
                )
            )

        return flowfileds
