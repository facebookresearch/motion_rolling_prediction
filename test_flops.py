# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import random
import time

import numpy as np
import torch

from fvcore.nn import flop_count_table, FlopCountAnalysis

from loguru import logger
from utils.constants import DataTypeGT

from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_args


def main():
    args = train_args()

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, diffusion = create_model_and_diffusion(args)

    device = "cuda" if not args.cpu else "cpu"
    model.to(device)
    logger.info(
        "Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )

    bs = 1
    num_feats = 132
    num_feats_sparse = 54

    timesteps = torch.tensor(
        list(range(args.input_motion_length)), device=device
    ).repeat(bs, 1)
    x_t = torch.randn((bs, args.input_motion_length, num_feats), device=device)
    x_start = torch.randn((bs, args.input_motion_length, num_feats), device=device)
    cond = {
        DataTypeGT.SPARSE: torch.randn(
            (bs, args.rolling_sparse_ctx + 1, num_feats_sparse), device=device
        ),
        DataTypeGT.MOTION_CTX: torch.randn(
            (bs, args.rolling_motion_ctx, num_feats), device=device
        ),
    }

    flops = FlopCountAnalysis(
        model,
        (
            x_t,
            timesteps,
            cond,
            x_start,
        ),
    )
    logger.info(f"{flops.by_operator()}")
    logger.info(f"{flops.by_module()}")
    logger.info(f"{flop_count_table(flops)}")
    logger.info(f"{flops.total() // 10**6} MFlops")

    # code for testing time per forward pass
    ITERATIONS = 1000
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(ITERATIONS):
            diffusion.ddim_sample(
                model,
                x_t,
                timesteps,
                cond,
                clip_denoised=False,
                model_kwargs={
                    "force_mask": False,  # conditional
                    "guidance": 1.0,  # no guidance
                    "x_start": x_start,
                },
            )
        end = time.time()
    ms_per_denoising = (end - start) / ITERATIONS * 1000
    logger.info(f"Time per denoising step: {ms_per_denoising:.2f} ms")
    logger.info(f"FPS: {1000 / ms_per_denoising:.2f}")


if __name__ == "__main__":
    main()
