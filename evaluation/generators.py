# Copyright (c) Meta Platforms, Inc. All Rights Reserved
from typing import Any, Dict, Optional, Tuple

import torch
import utils.constants as cnst

from loguru import logger

from numpy.linalg import norm

from utils import utils_transform

from utils.constants import DataTypeGT, ModelOutputType, RollingType


def create_generator(args, model, dataset, device):
    """
    Create a BaseGenerationWrapper from a library of pre-defined Generators.

    :param name: the name of the generator.
    """
    return RollingGenerationWrapper(
        args, model, dataset, device
    )


class BaseGenerationWrapper:
    def __init__(self, args, model, dataset, device):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.device = device

        self.no_normalization = args.no_normalization
        self.input_motion_length = args.input_motion_length

    def get_folder_suffix(self):
        suffix = ""
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

    def __init__(self, args, model, dataset, device):
        super().__init__(args, model, dataset, device)
        self.rolling_motion_ctx = args.rolling_motion_ctx
        self.rolling_sparse_ctx = args.rolling_sparse_ctx
        self.rolling_latency = args.rolling_latency

        logger.info(
            f"Rolling testing with {self.input_motion_length} frames."
        )

    def get_folder_suffix(self):
        suffix = super().get_folder_suffix()
        return "_rolling" + suffix

    def __call__(
        self,
        gt_data,
        sparse_original,
        return_intermediates=False,
        return_predictions=False,
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
            for i in range(ctx_margin):
                all_predictions[i] = self.inv_transform(
                    gt_data[0, i : i + self.input_motion_length].cpu()
                )

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

            result = self.model(x_start, cond, x_start=x_start)
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
