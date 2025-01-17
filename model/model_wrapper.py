import math

import numpy as np

import torch as th
import torch.nn as nn
from diffusion.gaussian_diffusion import _extract_into_tensor, GaussianDiffusion
from utils.constants import (
    DataTypeGT,
    ModelOutputType,
    PredictionInputType,
    PredictionTargetType,
)

ALL_FUNCTIONS = {
    PredictionTargetType.PCAF_COSINE: lambda num_timesteps: [
        1 - math.cos(t / num_timesteps * math.pi / 2)
        for t in range(1, num_timesteps + 1)
    ],
    PredictionTargetType.PCAF_COSINE_SQ: lambda num_timesteps: [
        1 - (math.cos(t / num_timesteps * math.pi / 2)) ** 2
        for t in range(1, num_timesteps + 1)
    ],
    PredictionTargetType.PCAF_LINEAR: lambda num_timesteps: [
        t / num_timesteps for t in range(1, num_timesteps + 1)
    ],
}


class ModelWrapper(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        target_type: PredictionTargetType,
        prediction_input_type: PredictionInputType,
    ):
        super().__init__()
        self.target_type = target_type
        self.prediction_input_type = prediction_input_type
        self.model = model
        self.diffusion = diffusion
        self.DEFAULT_NUM_BETAS = 16

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def convert_to_fp16(self):
        self.model.convert_to_fp16()

    def transform_model_output(self, model_output, cond, t, x_start=None):
        """
        The model_output is the clean motion. This method implements several transformations in the output of the
        model, depending on the target_type. Options:
        - POSITIONS: the model output is the clean motion (standard diffusion)
        - POSITIONS_SNR: the model output is capped by how much the motion can change in the current timestep. The
        earlier in the diffusion process, the more the model can change the motion. The rate of change is defined by
        the SNR of the noise at the current timestep.
        - POSITIONS_SIGMA: same as POSITIONS_SNR, but the rate of change is defined by the sigma of the noise.
        - POSITIONS_SIGMA2: same as POSITIONS_SNR, but the rate of change is defined by twice the sigma of the noise.
        - POSITIONS_CONST: same as POSITIONS_SIGMA, but sigma is 1 for all timesteps.
        - REL_OFFSETS: the model output becomes the relative offset from the previous denoised pose.
        - ABS_OFFSETS: the model output becomes the absolute offset from the last context pose.
        """
        if self.target_type == PredictionTargetType.POSITIONS:
            return model_output
        elif (
            self.target_type == PredictionTargetType.POSITIONS_SIGMA
            or self.target_type == PredictionTargetType.POSITIONS_SIGMA2
            or self.target_type == PredictionTargetType.PCAF_COSINE
            or self.target_type == PredictionTargetType.PCAF_COSINE_SQ
            or self.target_type == PredictionTargetType.PCAF_LINEAR
        ):
            assert x_start is not None, "x_start is required for POSITIONS_SIGMA"
            if (
                self.target_type == PredictionTargetType.POSITIONS_SIGMA
                or self.target_type == PredictionTargetType.POSITIONS_SIGMA2
            ):
                std = self.diffusion.get_sigma_from_t(t, t.shape)
                if self.target_type == PredictionTargetType.POSITIONS_SIGMA2:
                    std = 2 * std
            else:
                uncertainty = ALL_FUNCTIONS[self.target_type](
                    self.diffusion.num_timesteps
                )
                std = _extract_into_tensor(np.array(uncertainty), t, t.shape)
            return x_start + std.unsqueeze(-1) * th.tanh(model_output - x_start)
        elif self.target_type == PredictionTargetType.POSITIONS_CONST:
            assert x_start is not None, "x_start is required for POSITIONS_CONST"
            return x_start + th.tanh(model_output - x_start)
        else:
            raise NotImplementedError

    def forward(self, x, t, cond, x_start=None, **kwargs):
        """
        x: noisy motion
        t: timestep
        cond: conditioning
        x_start: clean input motion (a.k.a. previous prediction)
        """
        if self.prediction_input_type == PredictionInputType.NONE:
            x = th.zeros_like(x)
        elif self.prediction_input_type == PredictionInputType.LAST_GENERATED:
            last_gen_pose = cond[DataTypeGT.MOTION_CTX][:, -1:]
            x = last_gen_pose.repeat(1, x.shape[1], 1)
        elif self.prediction_input_type == PredictionInputType.CLEAN:
            x = x_start.float()

        model_output = self.model(x, t, cond, **kwargs)
        assert isinstance(model_output, dict), "model output must be a dict"
        assert (
            ModelOutputType.RELATIVE_ROTS in model_output.keys()
        ), "RELATIVE_ROTS must be in model output"
        relative_rots = model_output[ModelOutputType.RELATIVE_ROTS]
        bs, nframes = relative_rots.shape[:2]
        model_output[ModelOutputType.RELATIVE_ROTS] = self.transform_model_output(
            relative_rots, cond, t, x_start
        )
        return model_output
