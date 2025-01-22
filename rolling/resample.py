# MIT License
# Copyright (c) 2021 OpenAI
#
# This code is based on https://github.com/openai/guided-diffusion
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch as th
from loguru import logger
from utils.constants import RollingType
from typing import Optional


@dataclass
class NoiseScheduleData:
    timesteps: th.Tensor
    weights: th.Tensor
    padding_mask: Optional[th.Tensor]


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

    def __init__(self, pred_length):
        self.pred_length = pred_length
        self._weights = np.ones([self.pred_length])

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
        indices = th.round(
            th.linspace(0, self.pred_length - 1, seq_len)
        ).long()
        indices = indices.unsqueeze(0).repeat((batch_size, 1))
        weights = th.ones_like(indices)

        return NoiseScheduleData(indices.to(device), weights.to(device), None)



def create_named_schedule_sampler(
    name: RollingType, pred_length: int
) -> ScheduleSampler:
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == RollingType.UNIFORM:
        logger.info("Using standard time schedule.")
        return UniformSampler(pred_length)
    elif name == RollingType.ROLLING:
        logger.info("Using rolling time schedule.")
        return RollingSampler(pred_length)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")
