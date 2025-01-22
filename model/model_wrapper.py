import math

import numpy as np

import torch as th
import torch.nn as nn
from utils.constants import (
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


def _extract_into_tensor_rolling(arr, t, broadcast_shape):
    bs, sl = t.shape
    res = th.take_along_dim(
        th.from_numpy(arr).to(device=t.device), t.reshape(-1), dim=-1
    ).reshape(bs, sl)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if len(timesteps.shape) == 1:
        res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    else:
        return _extract_into_tensor_rolling(arr, timesteps, broadcast_shape)


class ModelWrapper(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        target_type: PredictionTargetType,
        prediction_input_type: PredictionInputType,
        prediction_length: int,
    ):
        super().__init__()
        self.target_type = target_type
        self.prediction_input_type = prediction_input_type
        self.model = model
        self.prediction_length = prediction_length
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

    def transform_model_output(self, model_output, t, prev_pred=None):
        """
        The model_output is the clean motion. This method implements several transformations in the output of the
        model, depending on the target_type.
        """
        if self.target_type == PredictionTargetType.POSITIONS:
            return model_output
        elif (
            self.target_type == PredictionTargetType.PCAF_COSINE
            or self.target_type == PredictionTargetType.PCAF_COSINE_SQ
            or self.target_type == PredictionTargetType.PCAF_LINEAR
        ):
            assert prev_pred is not None, "prev_pred is required for PCAF reparameterization"
            uncertainty = ALL_FUNCTIONS[self.target_type](self.prediction_length)
            uncertainty = _extract_into_tensor(np.array(uncertainty), t, t.shape).unsqueeze(-1)
            return prev_pred + uncertainty * th.tanh(model_output - prev_pred)
        else:
            raise NotImplementedError

    def q_sample(self, x_start, t):
        """
        Diffuse the input x_start by t steps.
        """
        # TODO add q_sample logic here
        raise NotImplementedError

    def forward(self, prev_pred, t, cond, **kwargs):
        """
        prev_pred: previous prediction
        t: timestep
        cond: conditioning
        """
        if self.prediction_input_type == PredictionInputType.NONE:
            model_input = th.zeros_like(prev_pred)
        elif self.prediction_input_type == PredictionInputType.CLEAN:
            model_input = prev_pred
        elif self.prediction_input_type == PredictionInputType.NOISY:
            model_input = self.q_sample(prev_pred, t)
        else:
            raise NotImplementedError

        model_output = self.model(model_input.float(), t, cond, **kwargs)
        assert isinstance(model_output, dict), "model output must be a dict"
        assert (
            ModelOutputType.RELATIVE_ROTS in model_output.keys()
        ), "RELATIVE_ROTS must be in model output"
        relative_rots = model_output[ModelOutputType.RELATIVE_ROTS]
        model_output[ModelOutputType.RELATIVE_ROTS] = self.transform_model_output(
            relative_rots, t, prev_pred
        )
        return model_output
