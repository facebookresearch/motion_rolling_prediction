"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

from typing import Final, Optional

import numpy as np

# MIT License
# Copyright (c) 2021 OpenAI
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch as th
import utils.constants as constants
from data_loaders.dataloader import TrainDataset
from diffusion.gaussian_diffusion import (
    _extract_into_tensor,
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from evaluation.utils import BodyModelsWrapper
from torch.distributions.exponential import Exponential
from utils.constants import (
    DataTypeGT,
    FreeRunningJumpType,
    LossDistType,
    ModelOutputType,
    MotionLossType,
    PredictionTargetType,
    RollingType,
)
from utils.utils_transform import sixd2aa


def loss_distance(
    a: th.Tensor,
    b: th.Tensor,
    dist_type: LossDistType = LossDistType.L2,
    joint_dim: int = 6,
):
    if dist_type == LossDistType.L1:
        p = 1
    elif dist_type == LossDistType.L2:
        p = 2
    else:
        raise ValueError(f"Unknown loss type {dist_type}")

    bs, n, c = a.shape
    loss = (
        th.norm(
            (a - b).reshape(-1, joint_dim),
            p,
            1,
        )
        .reshape(bs, n, c // joint_dim)
        .mean(dim=-1)
    )
    return loss


def loss_velocity(
    a: th.Tensor,
    b: th.Tensor,
    dist_type: LossDistType = LossDistType.L2,
    joint_dim: int = 6,
):
    bs, n, c = a.shape
    loss = th.zeros((bs, n), device=a.device)
    a_vel = a[:, 1:] - a[:, :-1]
    b_vel = b[:, 1:] - b[:, :-1]
    loss[:, :-1] = loss_distance(a_vel, b_vel, dist_type, joint_dim)
    return loss


def loss_jitter(a: th.Tensor):
    # Loss penalizing high jitter.
    bs, n, c = a.shape

    loss = th.zeros((bs, n), device=a.device)
    # we compute the derivative of the acceleration --> jerk
    jitter = a[:, 3:] - 3 * a[:, 2:-1] + 3 * a[:, 1:-2] - a[:, :-3]
    # we compute the norm of the derivative of the jerk per joint (6d rot), and average over joints
    loss[:, 3:] = (
        th.norm(
            jitter.reshape(-1, 6),
            2,
            1,
        )
        .reshape(bs, n - 3, c // 6)
        .mean(dim=-1)
    )
    return loss


class CustomBaseGaussianDiffusion(GaussianDiffusion):
    def __init__(
        self,
        **kwargs,
    ):
        super(CustomBaseGaussianDiffusion, self).__init__(
            **kwargs,
        )
        self.loss_dist_type = kwargs.get("loss_dist_type", LossDistType.L2)
        self.loss_velocity = kwargs.get("loss_velocity", 0.0)
        self.loss_jitter = kwargs.get("loss_jitter", 0.0)
        self.loss_fk = kwargs.get("loss_fk", 0.0)
        self.loss_fk_vel = kwargs.get("loss_fk_vel", 0.0)
        if (
            self.loss_fk > 0
            or self.loss_fk_vel > 0
        ):
            support_dir = kwargs.get("support_dir", None)
            assert support_dir is not None, "Support dir is required for FK loss"
            self.body_model = BodyModelsWrapper(support_dir)
        assert (
            self.model_mean_type == ModelMeanType.START_X
        ), "Only X_start pred is supported for RDM"

    def process_prediction_through_fk(
        self,
        output: dict,
        gt_data: dict,
        body_model: BodyModelsWrapper,
        dataset: TrainDataset,
    ):
        gender_fk = gt_data[DataTypeGT.SMPL_GENDER][0]
        model_type = gt_data[DataTypeGT.SMPL_MODEL_TYPE][0]
        local_rots = dataset.inv_transform(output[ModelOutputType.RELATIVE_ROTS])
        bs, seq_len = local_rots.shape[:2]
        local_rots_aa = sixd2aa(local_rots.reshape(-1, 6)).float().reshape(-1, 66)
        body_model = body_model.to(local_rots_aa.device)

        body_params = {
            "root_orient": local_rots_aa[:, :3],
            "pose_body": local_rots_aa[:, 3:66],
        }
        if (
            ModelOutputType.SHAPE_PARAMS in output
            and output[ModelOutputType.SHAPE_PARAMS] is not None
        ):
            # if shape params are predicted, use them
            body_params["betas"] = (
                output[ModelOutputType.SHAPE_PARAMS]
                .repeat(1, seq_len, 1)
                .reshape(-1, 16)
            )  # [bs * seq_len, 16]
        elif DataTypeGT.SHAPE_PARAMS in gt_data:
            # if shape params are not predicted, use the GT
            body_params["betas"] = gt_data[DataTypeGT.SHAPE_PARAMS].reshape(
                -1, 16
            )  # [bs * seq_len, 16]
        # if shape params are not in GT and not predicted, use default shape
        body_pose = body_model.grad_fk(
            body_params,
            model_type,
            gender_fk,
        )
        pred_joints = body_pose.Jtr[:, : constants.NUM_JOINTS_SMPL].reshape(
            bs, seq_len, constants.NUM_JOINTS_SMPL, 3
        )
        gt_joints = gt_data[DataTypeGT.WORLD_JOINTS]
        assert (
            pred_joints.shape == gt_joints.shape
        ), f"{pred_joints.shape} != {gt_joints.shape}"
        # make them relative to the head as in HMD-Poser + add GT head translation
        pred_joints = (
            pred_joints
            - pred_joints[:, :, [constants.HEAD_JOINT_IDX]]
            + gt_joints[:, :, [constants.HEAD_JOINT_IDX]]
        )
        output[ModelOutputType.WORLD_JOINTS] = pred_joints
        return output

    def p_mean_variance(
        self,
        model,
        x,
        t,
        cond,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Custom p_mean_variance function, which is used for our models. The difference is that we return
        the dictionary of model outputs, instead of just the output of the diffusion process.
        This way, we can add additional heads outside the diffusion paradigm (e.g. for shape regression).
        """
        B, C = x.shape[:2]
        assert t.shape == (B,) or t.shape == (B, C)
        if model_kwargs is not None:
            model_output = model(x, self._scale_timesteps(t), cond, **model_kwargs)
        else:
            model_output = model(x, self._scale_timesteps(t), cond)

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        pred_xstart = model_output[
            ModelOutputType.RELATIVE_ROTS
        ]  # RELATIVE_ROTS is the output of the diffusion process
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        out_dict = {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
        pred_xstart = model_output[ModelOutputType.RELATIVE_ROTS]
        # add additional outputs to the dictionary
        for k in model_output:
            out_dict[k] = model_output[k]
        return out_dict


class DiffusionModel(CustomBaseGaussianDiffusion):
    def __init__(
        self,
        **kwargs,
    ):
        super(DiffusionModel, self).__init__(
            **kwargs,
        )
        assert self.loss_type == LossType.MSE, "only MSE loss is supported"

    def training_losses(
        self, model, gt_data, t, cond, model_kwargs=None, noise=None, dataset=None
    ):
        x_start = gt_data[DataTypeGT.RELATIVE_ROTS]
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        output = model(x_t, self._scale_timesteps(t), cond, **model_kwargs)

        target = x_start

        terms = {}
        terms[MotionLossType.LOSS] = 0
        terms[MotionLossType.ROT_MSE] = loss_distance(
            target,
            output[ModelOutputType.RELATIVE_ROTS],
            dist_type=self.loss_dist_type,
        ).mean(dim=-1)
        terms[MotionLossType.LOSS] += terms[MotionLossType.ROT_MSE]
        if self.loss_velocity != 0:
            terms[MotionLossType.VEL_MSE] = loss_velocity(
                target,
                output[ModelOutputType.RELATIVE_ROTS],
                dist_type=self.loss_dist_type,
            ).mean(dim=-1)
            terms[MotionLossType.LOSS] += (
                terms[MotionLossType.VEL_MSE] * self.loss_velocity
            )
        if self.loss_jitter != 0:
            terms[MotionLossType.JITTER] = loss_jitter(
                output[ModelOutputType.RELATIVE_ROTS],
            ).mean(dim=-1)
            terms[MotionLossType.LOSS] += (
                terms[MotionLossType.JITTER] * self.loss_jitter
            )
        if self.loss_fk != 0 or self.loss_fk_vel > 0:
            assert (
                dataset is not None
            ), "Dataset is required for FK loss (inv transform)"
            output = self.process_prediction_through_fk(
                output, gt_data, self.body_model, dataset=dataset
            )
            bs, seq_len = output[ModelOutputType.WORLD_JOINTS].shape[:2]
            pred_joints = output[ModelOutputType.WORLD_JOINTS].reshape(
                (bs, seq_len, -1)
            )
            gt_joints = gt_data[DataTypeGT.WORLD_JOINTS].reshape((bs, seq_len, -1))
            if self.loss_fk != 0:
                terms[MotionLossType.JOINTS_MSE] = loss_distance(
                    pred_joints,
                    gt_joints,
                    dist_type=self.loss_dist_type,
                    joint_dim=3,
                ).mean(dim=-1)
                terms[MotionLossType.LOSS] += (
                    terms[MotionLossType.JOINTS_MSE] * self.loss_fk
                )
        return terms


class RollingDiffusionModel(CustomBaseGaussianDiffusion):
    """
    Simplified version of GaussianDiffusion, which is used for rolling diffusion
    """

    def __init__(
        self,
        **kwargs,
    ):
        super(RollingDiffusionModel, self).__init__(
            **kwargs,
        )
        self.rolling_type = kwargs["rolling_type"]
        assert (
            self.rolling_type != RollingType.UNIFORM
        ), "Uniform schedule for RDM is not supported."
        self.motion_cxt_len = kwargs.get("rolling_motion_ctx", 10)
        self.sparse_cxt_len = kwargs.get("rolling_sparse_ctx", 10)
        self.lat = kwargs.get("rolling_latency", 0)
        self.max_freerunning_steps = kwargs.get("rolling_fr_frames", 0)
        self.freerunning_jump = kwargs.get("rolling_fr_jump", FreeRunningJumpType.NONE)
        self.ctx_perturbation = kwargs.get("ctx_perturbation", 0.0)
        self.sp_perturbation = kwargs.get("sp_perturbation", 0.0)
        self.mask_cond_fn = kwargs.get("mask_cond_fn", None)
        self.target_type = kwargs.get("target_type", PredictionTargetType.POSITIONS)
        self.clamp_noise = kwargs.get("clamp_noise", False)

    def get_freerunning_jump(self, max_frames):
        if self.freerunning_jump == FreeRunningJumpType.NONE:
            # no jump (just go to the next frame)
            return 1
        elif self.freerunning_jump == FreeRunningJumpType.UNIFORM:
            # random jump between 1 and max_frames following a uniform distribution
            return th.randint(1, max_frames, (1,))[0].item()
        elif self.freerunning_jump == FreeRunningJumpType.EXPONENTIAL:
            # random jump between 1 and max_frames following an exponential distribution with P(x<max_frames) = 0.999
            LN_PROB = 6.9077552789  # result of -ln(1-0.999)
            lambda_ = th.tensor(LN_PROB / max_frames)
            sampled = Exponential(lambda_, (1,)).sample()
            return min(max(1, sampled.ceil().int().item()), max_frames)
        else:
            raise NotImplementedError

    def perturb_context(self, context):
        if self.ctx_perturbation > 0.0:
            context = context + th.randn_like(context) * self.ctx_perturbation
        return context

    def perturb_sparse(self, sparse):
        # The std value for each batch element is sampled from 0 to this maximum std.
        if self.sp_perturbation > 0.0:
            # first decide random std for each batch element
            std = (
                th.rand(sparse.shape[0], device=sparse.device).float()
                * self.sp_perturbation
            )
            # then add noise to sparse tracking signal according to std
            while len(std.shape) < len(sparse.shape):
                std = std.unsqueeze(-1)
            sparse = sparse + th.randn_like(sparse) * std
        return sparse

    def freerunning_step(
        self, model, i, x_start, cond, t, model_kwargs, update_context=True
    ):
        nframes = t.shape[1]
        x_start_ = x_start[:, i : i + nframes]
        x_start_[:, -1] = th.zeros_like(
            x_start_[:, -1]
        )  # we set the new frame of the long-term future to 0. This is done to let the network know which is the initial prediction, and be able to reduce uncertainty.
        # x_start_[:, -1] = x_start[
        #    :, -2
        # ]  # we duplicate the last frame as it's going to be done at inference stage. This is done because q_sample doesn't fully destroy the signal, and we don't want to leak GT.
        cond_ = {
            DataTypeGT.SPARSE: cond[DataTypeGT.SPARSE][
                :, i : i + self.sparse_cxt_len + 1 + self.lat
            ],
            DataTypeGT.MOTION_CTX: cond[DataTypeGT.MOTION_CTX][
                :, i : i + self.motion_cxt_len
            ],
        }

        x_t = self.q_sample(x_start_, t, clamp_noise=self.clamp_noise).float()
        model_kwargs["x_start"] = x_start_
        model_output = model(x_t, t, cond_, **model_kwargs)

        if update_context:
            x_start[:, i : i + nframes] = model_output[ModelOutputType.RELATIVE_ROTS]
            cond[DataTypeGT.MOTION_CTX][:, i + self.motion_cxt_len] = (
                self.perturb_context(model_output[ModelOutputType.RELATIVE_ROTS][:, 0])
            )
        return model_output

    def run_freerunning(self, model, gt_data, t, cond, model_kwargs=None):
        nframes = t.shape[1]
        x_start = gt_data[
            DataTypeGT.RELATIVE_ROTS
        ].clone()  # we initialize noisy prediction with GT for the first iteration.
        x_start[:, nframes:] = 0.0  # we set the future to 0
        cond[DataTypeGT.MOTION_CTX] = cond[DataTypeGT.MOTION_CTX].clone()

        fr = th.randint(0, self.max_freerunning_steps + 1, (1,))[0].item()
        no_grad_steps, grad_steps = fr, 1

        t_ = t
        # slice gt_data so that it matches the motion segment predicted after the FreeRunning frames
        s0 = fr # start timeframe where we start to compute loss
        s1 = fr + nframes  # end timeframe where we stop computing loss
        gt_data = {
            k: (
                gt_data[k][:, s0:s1]
                if isinstance(gt_data[k], th.Tensor) and len(gt_data[k].shape) >= 3
                else gt_data[k]
            )
            for k in gt_data.keys()
        }

        i = 0
        # NO GRAD stage
        model.eval()
        with th.no_grad():
            while i < no_grad_steps:
                self.freerunning_step(
                    model, i, x_start, cond, t_, model_kwargs, update_context=True
                )
                # increase i
                i += self.get_freerunning_jump(nframes)
                i = min(i, no_grad_steps)  # can jump up to no_grad_steps
        model.train()
        # GRAD stage
        while i < no_grad_steps + grad_steps:
            update = i != no_grad_steps + grad_steps - 1
            model_output = self.freerunning_step(
                model,
                i,
                x_start,
                cond,
                t_,
                model_kwargs,
                update_context=update,
            )

            # increase i
            if i == no_grad_steps + grad_steps - 1:
                i += 1
            else:
                # skip random frames from 1 to nframes, but can't go beyond no_grad_steps + grad_steps - 1
                # the -1 is to make sure the last denoising only does a step of size 1.
                i += self.get_freerunning_jump(nframes)
                i = min(i, no_grad_steps + grad_steps - 1)

        last_ctx_frame = i - 1 + self.motion_cxt_len
        prev_pred = x_start[:, i - 1 : i - 1 + nframes]
        return model_output, gt_data, prev_pred, last_ctx_frame

    def compute_losses(
        self,
        output: dict,
        gt_data: dict,
        cond: dict,
        t: th.Tensor,
        last_ctx_frame: int,
        prev_pred: th.Tensor,
        dataset: Optional[TrainDataset],
    ):
        """
        Computes the losses given:
        - output: the output dictionary of the model
        - gt_data: the ground truth dictionary
        - cond: the conditioning data dictionary
        - t: the timestep tensor
        - last_ctx_frame: the last context frame
        - prev_pred: the previous prediction
        """
        nframes = t.shape[1]
        loss_weights = self.get_loss_weights(t)

        terms = {}
        terms[MotionLossType.LOSS] = 0
        terms[MotionLossType.ROT_MSE] = loss_distance(
            gt_data[DataTypeGT.RELATIVE_ROTS],
            output[ModelOutputType.RELATIVE_ROTS],
            dist_type=self.loss_dist_type,
        )
        terms[MotionLossType.LOSS] += terms[MotionLossType.ROT_MSE] * loss_weights

        if self.loss_velocity != 0:
            terms[MotionLossType.VEL_MSE] = loss_velocity(
                gt_data[DataTypeGT.RELATIVE_ROTS],
                output[ModelOutputType.RELATIVE_ROTS],
                dist_type=self.loss_dist_type,
            )
            terms[MotionLossType.LOSS] += (
                terms[MotionLossType.VEL_MSE] * self.loss_velocity * loss_weights
            )
        if self.loss_fk != 0:
            assert (
                dataset is not None
            ), "Dataset is required for FK loss (inv transform)"
            output = self.process_prediction_through_fk(
                output, gt_data, self.body_model, dataset=dataset
            )
            bs, seq_len = output[ModelOutputType.WORLD_JOINTS].shape[:2]
            pred_joints = output[ModelOutputType.WORLD_JOINTS].reshape(
                (bs, seq_len, -1)
            )
            gt_joints = gt_data[DataTypeGT.WORLD_JOINTS].reshape((bs, seq_len, -1))
            if self.loss_fk != 0:
                terms[MotionLossType.JOINTS_MSE] = loss_distance(
                    pred_joints,
                    gt_joints,
                    dist_type=self.loss_dist_type,
                    joint_dim=3,
                )
                terms[MotionLossType.LOSS] += (
                    terms[MotionLossType.JOINTS_MSE] * self.loss_fk * loss_weights
                )
            if self.loss_fk_vel != 0:
                terms[MotionLossType.JOINTS_VEL_MSE] = loss_velocity(
                    pred_joints,
                    gt_joints,
                    dist_type=self.loss_dist_type,
                    joint_dim=3,
                )
                terms[MotionLossType.LOSS] += (
                    terms[MotionLossType.JOINTS_VEL_MSE]
                    * self.loss_fk_vel
                    * loss_weights
                )

        return terms

    def training_losses(
        self, model, gt_data, t, cond, model_kwargs=None, noise=None, dataset=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        cond[DataTypeGT.SPARSE] = self.mask_cond_fn(
            self.perturb_sparse(cond[DataTypeGT.SPARSE]), True
        )
        cond[DataTypeGT.MOTION_CTX] = self.perturb_context(cond[DataTypeGT.MOTION_CTX])
        if self.max_freerunning_steps > 0:
            model_output, gt_data, prev_pred, last_ctx_frame = self.run_freerunning(
                model, gt_data, t, cond, model_kwargs
            )
        else:
            # no freerunning
            x_start = gt_data[DataTypeGT.RELATIVE_ROTS]
            x_t = self.q_sample(x_start, t, clamp_noise=self.clamp_noise).float()
            model_kwargs["x_start"] = x_start
            model_output = model(x_t, t, cond, **model_kwargs)
            last_ctx_frame = self.motion_cxt_len
            prev_pred = x_start

        terms = self.compute_losses(
            model_output,
            gt_data,
            cond,
            t,
            last_ctx_frame,
            prev_pred,
            dataset,
        )
        # it returns dictionary of losses outputs with shape (bs, sl)
        return terms
