# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import json
import random

from pathlib import Path, PurePosixPath

import numpy as np

import torch
from data_loaders.dataloader import (
    get_dataloader,
    load_data,
    OnlineTrainDataset,
    TrainDataset,
)
from loguru import logger
from runner.training_loop import TrainLoop

from utils import dist_util

from utils.constants import DiffusionType
from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_args


def train_diffusion_model(args, dataloader, device="cuda"):
    logger.info("creating model and diffusion...")

    num_gpus = torch.cuda.device_count() if device != "cpu" else 1
    args.num_workers = args.num_workers * num_gpus

    model, diffusion = create_model_and_diffusion(args)

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
    
    if args.results_dir is None:
        raise FileNotFoundError("save_dir was not specified.")

    save_dir = Path(args.results_dir) / "checkpoints" / args.exp_name
    if save_dir.exists() and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(save_dir))
    elif not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as fw:
        # from Path to str
        serializable_args = {
            k: str(PurePosixPath(v)) if isinstance(v, Path) else v 
            for k, v in vars(args).items()
        }
        json.dump(serializable_args, fw, indent=4, sort_keys=True)

    logger.info("creating data loader...")
    dataset_data = load_data(
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
