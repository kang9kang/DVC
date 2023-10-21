import torch
import torch.nn as nn

from subnets.bitestimator import BitEstimator


class CompressionLoss(nn.Module):
    def __init__(self, warpweight, train_lambda):
        super(CompressionLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.warpweight = warpweight
        self.train_lambda = train_lambda

    def __new__(self, *args, **kwargs):
        if not hasattr(self, "_instance"):
            self._instance = super(CompressionLoss, self).__new__(self)
        return self._instance

    def forward(self, input_img, recon_image, warpframe, prediction, bpp):
        mse_loss = self.mse(recon_image, input_img)
        warploss = self.mse(warpframe, input_img)
        interloss = self.mse(prediction, input_img)

        distortion_loss = mse_loss + self.warpweight * (warploss + interloss)
        distribution_loss = bpp

        rd_loss = self.train_lambda * distortion_loss + distribution_loss

        psnr = 10 * torch.log10(1 / mse_loss)
        return rd_loss, psnr
