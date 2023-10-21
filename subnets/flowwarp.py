import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowWarpper(nn.Module):
    def __init__(self):
        super(FlowWarpper, self).__init__()
        self.backward_tensorgrid = [{} for i in range(4)]

    def forward(self, img_input, img_flow):
        device_id = img_input.device.index
        if str(img_flow.size()) not in self.backward_tensorgrid[device_id]:
            tensorHorizontal = (
                torch.linspace(-1.0, 1.0, img_flow.size(3))
                .view(1, 1, 1, img_flow.size(3))
                .expand(img_flow.size(0), -1, img_flow.size(2), -1)
            )
            tensorVertical = (
                torch.linspace(-1.0, 1.0, img_flow.size(2))
                .view(1, 1, img_flow.size(2), 1)
                .expand(img_flow.size(0), -1, -1, img_flow.size(3))
            )
            self.backward_tensorgrid[device_id][str(img_flow.size())] = (
                torch.cat([tensorHorizontal, tensorVertical], 1).cuda().to(device_id)
            )

        img_flow = torch.cat(
            [
                img_flow[:, 0:1, :, :] / ((img_input.size(3) - 1.0) / 2.0),
                img_flow[:, 1:2, :, :] / ((img_input.size(2) - 1.0) / 2.0),
            ],
            dim=1,
        )

        return F.grid_sample(
            input=img_input,
            grid=(
                self.backward_tensorgrid[device_id][str(img_flow.size())] + img_flow
            ).permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
