import torch
import torch.nn as nn
import torchac
import numpy as np
import math

from subnets import *


class VideoCompressor(nn.Module):
    def __init__(self):
        super(VideoCompressor, self).__init__()
        self.optical_flow = MESpynet()
        self.mv_encoder = AnalysisMvNet()
        self.mv_decoder = SynthesisMvNet()
        self.flowwarp = FlowWarpper()
        self.warpnet = WarpNet()
        self.resencoder = AnalysisNet()
        self.respriorencoder = AnalysisPriorNet()
        self.respriordecoder = SynthesisPriorNet()
        self.resdecoder = SynthesisNet()
        self.bitesimator_mv = BitEstimator(128)
        self.bitesimator_z = BitEstimator(64)

        self.mxrange = 150
        self.calrealbits = False

    def forward(self, img_input, img_ref):
        estmv = self.optical_flow(img_input, img_ref)
        mvfeature = self.mv_encoder(estmv)
        if self.training:
            quantized_mv = (
                mvfeature + torch.rand(mvfeature.shape).to(mvfeature.device) - 0.5
            )
        else:
            quantized_mv = torch.round(mvfeature)
        quantized_mv_upsample = self.mv_decoder(quantized_mv)
        prediction, warpframe = self.motioncopensation(img_ref, quantized_mv_upsample)

        input_residual = img_input - prediction
        feature_residual = self.resencoder(input_residual)
        z = self.respriorencoder(feature_residual)
        if self.training:
            compressed_z = z + torch.rand(z.shape).to(z.device) - 0.5
        else:
            compressed_z = torch.round(z)
        recon_sigma = self.respriordecoder(compressed_z)

        if self.training:
            compressed_feature_residual = (
                feature_residual + torch.rand(feature_residual.shape).to(z.device) - 0.5
            )
        else:
            compressed_feature_residual = torch.round(feature_residual)

        recon_residual = self.resdecoder(compressed_feature_residual)
        recon_image = recon_residual + prediction
        clip_recon_image = torch.clamp(recon_image, 0, 1)

        total_bits_feature, prob_feature = self.feature_probs_based_sigma(
            compressed_feature_residual, recon_sigma
        )
        total_bits_z, prob_z = self.estrate_bits(compressed_z, self.bitesimator_z)
        total_bits_mv, prob_mv = self.estrate_bits(quantized_mv, self.bitesimator_mv)

        bpp_feature = total_bits_feature / (
            img_input.shape[0] * img_input.shape[2] * img_input.shape[3]
        )
        bpp_z = total_bits_z / (
            img_input.shape[0] * img_input.shape[2] * img_input.shape[3]
        )
        bpp_mv = total_bits_mv / (
            img_input.shape[0] * img_input.shape[2] * img_input.shape[3]
        )
        bpp = bpp_feature + bpp_z + bpp_mv

        return clip_recon_image, recon_image, warpframe, prediction, bpp

    def motioncopensation(self, img_ref, mv):
        warpframe = self.flowwarp(img_ref, mv)
        inputfeature = torch.cat([warpframe, img_ref], dim=1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def feature_probs_based_sigma(self, feature, sigma):
        def getrealbitsg(x, gaussian):
            # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
            cdfs = []
            x = x + self.mxrange
            n, c, h, w = x.shape
            for i in range(-self.mxrange, self.mxrange):
                cdfs.append(gaussian.cdf(i - 0.5).view(n, c, h, w, 1))
            cdfs = torch.cat(cdfs, 4).cpu().detach()

            byte_stream = torchac.encode_float_cdf(
                cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True
            )

            real_bits = (
                torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()
            )

            sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

            return sym_out - self.mxrange, real_bits

        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        total_bits = torch.sum(
            torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50)
        )

        if self.calrealbits and not self.training:
            decodedx, real_bits = getrealbitsg(feature, gaussian)
            total_bits = real_bits

        return total_bits, probs

    def estrate_bits(self, x, bitestimator):
        def getrealbits(x):
            cdfs = []
            x = x + self.mxrange
            n, c, h, w = x.shape
            for i in range(-self.mxrange, self.mxrange):
                cdfs.append(
                    bitestimator(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1)
                )
            cdfs = torch.cat(cdfs, 4).cpu().detach()
            byte_stream = torchac.encode_float_cdf(
                cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True
            )

            real_bits = torch.sum(
                torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()
            )

            sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

            return sym_out - self.mxrange, real_bits

        prob = bitestimator(x + 0.5) - bitestimator(x - 0.5)
        total_bits = torch.sum(
            torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50)
        )

        if self.calrealbits and not self.training:
            decodedx, real_bits = getrealbits(x)
            total_bits = real_bits

        return total_bits, prob
