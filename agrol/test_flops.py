# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import random

import numpy as np
import torch

from fvcore.nn import flop_count_table, FlopCountAnalysis

from loguru import logger
from utils.config import pathmgr
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import sample_args


device = torch.device("cuda")


def load_diffusion_model(args):
    logger.info("Creating model and diffusion...")
    args.arch = args.arch[len("diffusion_") :]
    model, diffusion = create_model_and_diffusion(args)

    logger.info(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(pathmgr.get_local_path(args.model_path), map_location="cpu")
    load_model_wo_clip(model, state_dict)

    model.to("cpu")  # dist_util.dev())
    model.eval()  # disable random masking
    return model, diffusion


def main():
    args = sample_args()

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, diffusion = load_diffusion_model(args)

    bs = 1
    flops = FlopCountAnalysis(
        model,
        (
            torch.randn((bs, 196, 132)),
            torch.randint(0, 100, (bs,)),
            torch.randn((bs, 196, 54)),
        ),
    )
    logger.info(f"{flops.by_operator()}")
    logger.info(f"{flops.by_module()}")
    logger.info(f"{flop_count_table(flops)}")
    logger.info(f"{flops.total() // 10**6} MFlops")


if __name__ == "__main__":
    main()
