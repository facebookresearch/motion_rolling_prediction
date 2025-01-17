import torch
import torch.nn as nn


# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
# It also implements Interval-based CFG from
# https://arxiv.org/pdf/2404.07724
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, diffusion, min_snr: float, max_snr: float):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.diffusion = diffusion
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.interval_cfg = self.min_snr != -float("inf") or self.max_snr != float(
            "inf"
        )

    def forward(self, x, timesteps, cond, force_mask=False, guidance=1.0, **kwargs):
        if force_mask or guidance == 0.0:
            return self.model(x, timesteps, cond, force_mask=True, **kwargs)
        elif guidance == 1.0:
            return self.model(x, timesteps, cond, **kwargs)
        elif self.interval_cfg:
            snr = self.diffusion.get_snr_from_t(timesteps, timesteps.shape)
            mask_cfg = (snr >= self.min_snr) & (snr <= self.max_snr)
            guidance = torch.where(mask_cfg, guidance, 1.0).unsqueeze(-1)

        out = self.model(x, timesteps, cond, **kwargs)
        out_uncond = self.model(x, timesteps, cond, force_mask=True, **kwargs)
        return out_uncond + (guidance * (out - out_uncond))
