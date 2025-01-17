# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import json
import os
import random

import tempfile

import numpy as np

import torch
from data_loaders.dataloader import (
    get_dataloader,
    load_data_from_manifold,
    OnlineTrainDataset,
    TrainDataset,
)
from loguru import logger
from runner.training_loop import TrainLoop

from utils import dist_util

from utils.config import pathmgr
from utils.constants import DiffusionType
from utils.model_util import create_model_and_diffusion, get_model_class
from utils.parser_util import train_args


def train_diffusion_model(args, dataloader, device="cuda"):
    logger.info("creating model and diffusion...")
    # args.arch = args.arch[len("diffusion_") :]

    num_gpus = torch.cuda.device_count() if device != "cpu" else 1
    args.num_workers = args.num_workers * num_gpus

    model_cls = get_model_class(args)
    model, diffusion = create_model_and_diffusion(
        args,
        model_cls=model_cls,
    )

    if num_gpus > 1 and device != "cpu":
        logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
        dist_util.setup_dist()
        model = torch.nn.DataParallel(model).cuda()
        logger.info(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.module.parameters()) / 1000000.0)
        )
    else:
        dist_util.setup_dist(args.device)
        model.to(dist_util.dev() if device != "cpu" else "cpu")
        logger.info(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )

    logger.info("Loading training dependencies...")
    trainer = TrainLoop(args, model, diffusion, dataloader, device=device)
    logger.info("Training:")
    trainer.run_loop()
    logger.info("Done.")


def main():
    args = train_args()
    device = "cuda" if not args.cpu else "cpu"
    import torch.multiprocessing

    # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
    torch.multiprocessing.set_sharing_strategy("file_system")

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = os.path.join(args.results_dir, "checkpoints", args.exp_name)
    if save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif pathmgr.exists(save_dir) and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(save_dir))
    elif not pathmgr.exists(save_dir):
        pathmgr.mkdirs(save_dir)
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_args_path = os.path.join(tmpdirname, "args.json")
        args_path = os.path.join(save_dir, "args.json")
        with open(tmp_args_path, "w") as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)
        pathmgr.copy(
            src_path=tmp_args_path, dst_path=args_path, overwrite=True
        )  # copy to manifold

    logger.info("creating data loader...")
    dataset_data = load_data_from_manifold(
        args.dataset,
        args.dataset_path,
        "train",
        total_length=args.input_motion_length
        + max(args.rolling_motion_ctx, args.rolling_sparse_ctx)
        + args.rolling_fr_frames,
        max_samples=args.dataset_max_samples,
    )
    dataset_cls = (
        OnlineTrainDataset
        if args.diffusion_type in DiffusionType.ROLLING
        else TrainDataset
    )
    dataset = dataset_cls(
        args.dataset,
        dataset_data,
        args.input_motion_length,
        args.train_dataset_repeat_times,
        args.no_normalization,
        sparse_context=args.rolling_sparse_ctx,
        motion_context=args.rolling_motion_ctx,
        latency=args.rolling_latency,
        freerunning_frames=args.rolling_fr_frames,
        use_real_input=args.use_real_input,
        input_conf_threshold=args.input_conf_threshold,
    )
    dataloader = get_dataloader(
        dataset,
        "train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # args.lr_anneal_steps = (
    #    args.lr_anneal_steps // args.train_dataset_repeat_times
    # ) * len(
    #    dataloader
    # )  # the input lr_anneal_steps is by epoch, here convert it to the number of steps

    train_diffusion_model(args, dataloader, device=device)


if __name__ == "__main__":
    main()
