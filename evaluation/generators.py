# Copyright (c) Meta Platforms, Inc. All Rights Reserved
from typing import Any, Dict, Optional, Tuple

import torch
import utils.constants as cnst

from rolling.resample import create_named_schedule_sampler
from evaluation.simplify_loc2rot import joints2smpl

from loguru import logger

from numpy.linalg import norm

from utils import utils_transform

from utils.constants import DataTypeGT, ModelOutputType


def run_ik_on_tracking_input(sparse_input_vec, body_model, betas, device):
    """
    Runs IK to sample a pose that matches the sparse input vector (head + hands).
    - sparse_input_vec: [batch_size, features_length]
    """
    BS = sparse_input_vec.shape[0]
    body_model = body_model.to(device)
    j2s = joints2smpl(
        body_model.get_smplify_fn(),
        device=device,
        num_betas=cnst.NUM_BETAS_SMPL,
        num_joints=cnst.NUM_JOINTS_SMPL,
    )
    # logger.info("Running SMPLify, it may take a few minutes.")

    # We fill the position of the joints we have.
    joints_pos = torch.zeros((BS, cnst.NUM_JOINTS_SMPL, 3))
    joints_pos[:, cnst.HEAD_JOINT_IDX] = sparse_input_vec[:, cnst.INPUT_HEAD_IDCES]
    joints_pos[:, cnst.RHAND_JOINT_IDX] = sparse_input_vec[:, cnst.INPUT_RHAND_IDCES]
    joints_pos[:, cnst.LHAND_JOINT_IDX] = sparse_input_vec[:, cnst.INPUT_LHAND_IDCES]

    # We fill the confidence of each joint
    confidence_input = torch.zeros(cnst.NUM_JOINTS_SMPL)
    for idx in (cnst.HEAD_JOINT_IDX, cnst.RHAND_JOINT_IDX, cnst.LHAND_JOINT_IDX):
        confidence_input[idx] = 1

    # We run the IK
    pose, joints = j2s.joint2smpl(joints_pos, confidence_input, betas)

    # Transform to 6D representation
    gt_rotations_aa = torch.Tensor(pose[:, :66]).reshape(-1, 3)
    init_pose = utils_transform.aa2sixd(gt_rotations_aa).reshape(
        pose.shape[0], -1
    )  # [bs, f1] filled with result of an IK to match tracking signal

    return init_pose


def create_generator(args, model, dataset, device, body_model):
    """
    Create a BaseGenerationWrapper from a library of pre-defined Generators.

    :param name: the name of the generator.
    """
    return RollingGenerationWrapper(
        args, model, dataset, device, body_model
    )


class BaseGenerationWrapper:
    def __init__(self, args, model, dataset, device, body_model):
        self.args = args
        self.init_ik = getattr(args, "init_ik", False)
        self.model = model
        self.dataset = dataset
        self.device = device

        self.uncond = getattr(args, "uncond", False)

        self.no_normalization = args.no_normalization
        self.input_motion_length = args.input_motion_length

    def get_folder_suffix(self):
        suffix = ""
        if self.uncond:
            suffix += "_uncond"
        if self.dataset.eval_gap_config is not None:
            suffix += f"_{self.dataset.eval_gap_config}"
        return suffix

    def __call__(self, gt_data, sparse_original, return_intermediates=False):
        raise NotImplementedError

    def transform(self, data):
        return self.dataset.transform(data) if not self.no_normalization else data

    def inv_transform(self, data):
        return self.dataset.inv_transform(data) if not self.no_normalization else data

    def transform_to_aanorm(self, pred):
        # pred is [seq, 135] or [bs, seq, 135]
        bs = 1 if len(pred.shape) == 2 else pred.shape[0]
        seq_len = pred.shape[-2]
        if seq_len == 0:
            return pred
        pred = self.inv_transform(pred)
        return norm(
            utils_transform.sixd2aa(pred.reshape(-1, 6))
            .reshape(bs, seq_len, -1, 3)
            .numpy(),
            axis=-1,
        )


class RollingGenerationWrapper(BaseGenerationWrapper):

    def __init__(self, args, model, dataset, device, body_model):
        super().__init__(args, model, dataset, device, body_model)
        self.rolling_type = args.rolling_type
        self.rolling_motion_ctx = args.rolling_motion_ctx
        self.rolling_sparse_ctx = args.rolling_sparse_ctx
        self.rolling_latency = args.rolling_latency

        self.rolling_horizon = getattr(args, "rolling_horizon", -1)
        if (
            self.rolling_horizon == -1
        ):  # if unspecified, use the input motion length as default
            self.rolling_horizon = self.input_motion_length
        self.schedule_sampler = create_named_schedule_sampler(
            self.rolling_type, args.input_motion_length
        )

        t = self.schedule_sampler.sample(
            1, self.rolling_horizon, device=self.device, train=False
        ).timesteps
        if t.shape[-1] != self.input_motion_length:
            # repeat the last noise value for the rest of the sequence
            t = torch.cat(
                [
                    t,
                    t[:, -1].repeat(1, self.input_motion_length - self.rolling_horizon),
                ],
                dim=-1,
            )
        self.t = t

        logger.info(
            f"Rolling testing with '{self.rolling_type}' mode and {self.rolling_horizon} frames."
        )

    def get_folder_suffix(self):
        suffix = super().get_folder_suffix()
        if self.rolling_horizon != -1:
            suffix += f"_rh{self.rolling_horizon}"
        return "_rolling" + suffix

    def __call__(
        self,
        gt_data,
        sparse_original,
        return_intermediates=False,
        return_predictions=False,
        body_model=None,
        betas=None,
        **kwargs,
    ) -> Tuple[Dict[ModelOutputType, Optional[torch.Tensor]], Dict[str, Any]]:
        """
        gt_data: [bs, seq, f1]
        sparse_original: [bs, seq, f2]
        """
        assert (
            len(gt_data.shape) == len(sparse_original.shape) == 3
        ), "[bs, seq, f] expected"
        gt_data = self.transform(gt_data).to(self.device).float().clone()
        sparse_original = sparse_original.to(self.device).float()

        ctx_margin = max(self.rolling_motion_ctx, self.rolling_sparse_ctx)
        BS, TOTAL_LENGTH = gt_data.shape[:2]
        if TOTAL_LENGTH < self.input_motion_length + ctx_margin:
            logger.error("too short for rolling test")
            exit()

        if self.init_ik:
            IK_PADDING = (
                ctx_margin + 30
            )  # 30 frames for RPM to fix the initial pose
            assert (
                body_model is not None and betas is not None
            ), "IK requires body model"
            init_tracking = sparse_original[:, 0]  # [bs, f2]
            # we pad the output and the sparse signal to accommodate for the margin
            output = torch.zeros(
                (BS, TOTAL_LENGTH + IK_PADDING, gt_data.shape[-1]), device=self.device
            )
            sparse_padding = torch.zeros(
                (BS, IK_PADDING, sparse_original.shape[-1]), device=self.device
            )
            sparse_original = torch.cat([sparse_padding, sparse_original], dim=1)
            sparse_original[:, :IK_PADDING] = init_tracking.unsqueeze(
                1
            )  # propagate first tracking signal to the padded region (a.k.a. simulated past)

            init_pose = run_ik_on_tracking_input(
                init_tracking, body_model, betas, self.device
            )  # [bs, f1]
            init_pose_norm = self.transform(init_pose.to("cpu")).to(
                self.device
            )  # [bs, f1]

            init_pose_norm = init_pose_norm.unsqueeze(1)  # [bs, 1, f1]
            output[:, :] = init_pose_norm  # to use it as motion context
            gt_data[:, :] = (
                init_pose_norm  # to use it as initial noisy prediction window (static pose to the future)
            )
            output_shape = torch.zeros(
                (BS, TOTAL_LENGTH + IK_PADDING, cnst.NUM_BETAS_SMPL)
            )
        else:
            output = torch.zeros_like(gt_data)
            output[:, :ctx_margin] = gt_data[:, :ctx_margin]
            output_shape = torch.zeros((BS, TOTAL_LENGTH, cnst.NUM_BETAS_SMPL))

        current_idx = ctx_margin

        all_info = {}
        intermediates = []
        all_predictions = None
        if return_predictions:
            assert BS == 1, "only support batch size 1 for now"
            # initialize tensor [seq_len, self.input_motion_length, f1]
            all_predictions = torch.zeros(
                (TOTAL_LENGTH, self.input_motion_length, gt_data.shape[-1]),
                device="cpu",
            )
            if self.init_ik:
                all_predictions = None
            else:
                for i in range(ctx_margin):
                    all_predictions[i] = self.inv_transform(
                        gt_data[0, i : i + self.input_motion_length].cpu()
                    )

        t_bs = self.t.repeat(BS, 1)
        x_start = gt_data[:, ctx_margin : ctx_margin + self.input_motion_length]
        while current_idx < output.shape[1]:
            cur_motion_ctx = output[
                :, current_idx - self.rolling_motion_ctx : current_idx
            ]
            cur_sparse = sparse_original[
                :,
                current_idx
                - self.rolling_sparse_ctx : current_idx
                + 1
                + self.rolling_latency,
            ]
            cond = {
                DataTypeGT.SPARSE: cur_sparse,
                DataTypeGT.MOTION_CTX: cur_motion_ctx,
            }

            result = self.model(x_start, t_bs, cond, x_start=x_start)
            x_start = result[ModelOutputType.RELATIVE_ROTS]
            if return_predictions and all_predictions is not None:
                all_predictions[current_idx] = self.inv_transform(x_start.cpu())[0]
            if return_intermediates:
                raise NotImplementedError()  # TODO now we don't have noisy sample anymore
                noisy_x_start = result["sample"]
                intermediates.append(
                    [
                        current_idx,
                        self.transform_to_aanorm(cur_motion_ctx.cpu()),
                        self.transform_to_aanorm(x_start.cpu()),
                        self.transform_to_aanorm(noisy_x_start.cpu()),
                    ]
                )

            # save first frame, which is fully denoised now
            output[:, current_idx : current_idx + 1] = x_start[:, :1]
            if (
                ModelOutputType.SHAPE_PARAMS in result
                and result[ModelOutputType.SHAPE_PARAMS] is not None
                and output_shape is not None
            ):
                output_shape[:, current_idx] = result[ModelOutputType.SHAPE_PARAMS]
            else:
                output_shape = None

            x_start[:, :-1] = x_start.clone()[:, 1:]
            # add zeros to last frame, as it is done during training
            x_start[:, -1] = 0.0

            current_idx += 1

        if self.init_ik:
            output = output[:, IK_PADDING:]  # remove padding we added
            output_shape = (
                output_shape[:, IK_PADDING:] if output_shape is not None else None
            )

        result_inv = self.inv_transform(output.cpu())
        result = {
            ModelOutputType.RELATIVE_ROTS: result_inv,
            ModelOutputType.SHAPE_PARAMS: output_shape,
        }
        if return_predictions and all_predictions is not None:
            all_info["raw_predictions"] = all_predictions
        if return_intermediates:
            all_info["intermediates"] = intermediates
            all_info["gt"] = self.transform_to_aanorm(gt_data.cpu())
            all_info["prediction"] = self.transform_to_aanorm(output.cpu())
        return result, all_info
