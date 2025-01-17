# MIT License
# Copyright (c) 2021 OpenAI
#
# This code is based on https://github.com/openai/guided-diffusion
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.distributed as dist
from diffusion.gaussian_diffusion import GaussianDiffusion
from loguru import logger
from utils.constants import RollingType


@dataclass
class NoiseScheduleData:
    timesteps: th.Tensor
    weights: th.Tensor
    padding_mask: th.Tensor


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, seq_len, device, **kwargs):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
                 - padding_mask: a padding mask with 1's indicating frames that are part of the padding. None if no padding.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return NoiseScheduleData(indices, weights, None)


class UniformSampler(ScheduleSampler):
    """
    Samples a single timestep for each batch element (standard diffusion)
    """

    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class RollingSampler(UniformSampler):
    """
    Sampler that allows implementing "Rolling Diffusion Models"
    Paper link: https://arxiv.org/abs/2402.09470
    In particular, it samples the same increasing sequence of timesteps for all batch elements.
    E.g.: diffusion steps = 5 --> [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    """

    def sample(self, batch_size, seq_len, device, **kwargs):
        # same for all batch
        starting_point = (
            (self.diffusion.num_timesteps // seq_len)
            if seq_len > 0
            else self.diffusion.num_timesteps - 1
        )
        indices = th.round(
            th.linspace(starting_point, self.diffusion.num_timesteps - 1, seq_len)
        ).long()
        indices = indices.unsqueeze(0).repeat((batch_size, 1))
        weights = th.ones_like(indices)

        return NoiseScheduleData(indices.to(device), weights.to(device), None)


class RollingSamplerV2(UniformSampler):
    """
    RollingSamplerV2 is RollingSampler starting at 0 instead of the "starting_point".
    This fixes the issue that one of the timesteps was repeated at the middle of the sequence.
    """

    def sample(self, batch_size, seq_len, device, **kwargs):
        indices = th.round(
            th.linspace(0, self.diffusion.num_timesteps - 1, seq_len)
        ).long()
        indices = indices.unsqueeze(0).repeat((batch_size, 1))
        weights = th.ones_like(indices)

        return NoiseScheduleData(indices.to(device), weights.to(device), None)


class OMPSampler(UniformSampler):
    """
    Sampler that allows to sample increasing sequences of timesteps with random depth
    for all batch elements. Timesteps beyond the random depth are set to the last timestep.
    E.g.: diffusion steps = 5 --> [[0, 2, 4, 4, 4], [0, 1, 2, 3, 4], [4, 4, 4, 4, 4], [0, 1, 3, 4, 4]]
    """

    def sample(self, batch_size, seq_len, device, train=True, **kwargs):
        assert seq_len > 1, "OMPSampler requires seq_len > 1"
        # random max seq for each batch size
        indices = th.zeros((batch_size, seq_len), device=device).long()
        weights = th.zeros((batch_size, seq_len), device=device).float()
        indices += self.diffusion.num_timesteps - 1
        for i in range(batch_size):
            _max_seq = np.random.randint(1, seq_len + 1) if train else seq_len
            starting_point = (self.diffusion.num_timesteps - 1) // _max_seq
            indices[i, :_max_seq] = th.round(
                th.linspace(starting_point, self.diffusion.num_timesteps - 1, _max_seq)
            ).long()
            weights[i, :_max_seq] = 1

        padding_mask = (1 - weights).bool()
        return NoiseScheduleData(indices, weights, padding_mask)


class DFSampler(UniformSampler):
    """
    Random sampling of timesteps within a sequence.
    It can be considered the sampling performed in the DiffusionForcing paper:
    https://arxiv.org/abs/2407.01392
    """

    def sample(self, batch_size, seq_len, device, **kwargs):
        # all random from 0 to self.diffusion.num_timesteps
        indices = th.randint(
            1, self.diffusion.num_timesteps, (batch_size, seq_len)
        ).long()
        weights = th.from_numpy(self.weights()).float().to(device)
        return NoiseScheduleData(indices.to(device), weights.to(device), None)


def create_named_schedule_sampler(
    name: RollingType, diffusion: GaussianDiffusion
) -> ScheduleSampler:
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == RollingType.UNIFORM:
        logger.info("Using standard time schedule.")
        return UniformSampler(diffusion)
    elif name == RollingType.ROLLING:
        logger.info("Using rolling time schedule.")
        return RollingSampler(diffusion)
    elif name == RollingType.ROLLING_0:
        logger.info("Using rolling time schedule (starting at 0).")
        return RollingSamplerV2(diffusion)
    elif name == RollingType.OMP:
        logger.info("Using OMP time schedule.")
        return OMPSampler(diffusion)
    elif name == RollingType.DIFFUSIONFORCING:
        logger.info(
            "Using DiffusionForcing time schedule (all timesteps random inside a sequence)."
        )
        return DFSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")
