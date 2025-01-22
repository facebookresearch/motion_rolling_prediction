# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import os
import random

import numpy as np
import torch

from data_loaders.dataloader import TestDataset
from evaluation.evaluation import EvaluatorWrapper
from evaluation.generators import create_generator

# see https://fb.workplace.com/groups/271514828183094/permalink/457581169576458/
from evaluation.utils import BodyModelsWrapper
from evaluation.visualization import VisualizerWrapper

from loguru import logger

from utils.model_util import load_rpm_model
from utils.parser_util import sample_args
from pathlib import Path


def make_args_retrocompatible(args):
    if args.rolling_context != -1:
        logger.warning(
            f"OLD FORMAT for rolling context length = {args.rolling_context}. This is deprecated."
        )
        # it means it was trained with old unified rolling context for sparse and motion
        args.rolling_motion_ctx = args.rolling_context
        args.rolling_sparse_ctx = args.rolling_context


def main():
    args = sample_args()
    make_args_retrocompatible(args)
    device = f"cuda:{args.device}" if not args.cpu else "cpu"

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("Loading dataset...")
    dataset = TestDataset(
        args.dataset,
        args.dataset_path,
        args.no_normalization,
        max_samples=args.dataset_max_samples,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        eval_gap_config=args.eval_gap_config,
        num_features=args.motion_nfeat,
        use_real_input=args.use_real_input,
        input_conf_threshold=args.input_conf_threshold,
        test_split=args.test_split,
    )
    logger.info("Loading model...")
    model, _ = load_rpm_model(args, device=device)

    exp_name = args.model_path.parts[-2]
    checkpoint_name = args.model_path.parts[-1].split(".")[0]
    subfolder_name = (
        checkpoint_name[6:] if checkpoint_name.startswith(
            "model") else checkpoint_name
    )
    body_model = BodyModelsWrapper(args.support_dir)

    generator = create_generator(args, model, dataset, device, body_model)
    suffix = generator.get_folder_suffix()
    subfolder_name += suffix
    name_results_folder = "results"
    if args.test_split != "test":
        name_results_folder += f"_{args.test_split}"
    if args.init_ik:
        name_results_folder += "_init_ik"
    if args.use_real_input:
        name_results_folder += f"_real_input_C{args.input_conf_threshold}"
    output_dir = args.results_dir / name_results_folder / exp_name / subfolder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.vis or args.vis_gt:
        visualizer = VisualizerWrapper(
            args, generator, dataset, body_model, device)
        if args.vis:
            logger.info("Visualizing the results...")
            # visualizer.visualize_all(output_dir)
            for i in range(args.vis_reps):
                # set seed to i
                random.seed(args.seed + i)
                np.random.seed(args.seed + i)
                torch.manual_seed(args.seed + i)
                # visualize the test subset of the dataset
                visualizer.visualize_subset(
                    output_dir,
                    overwrite=args.vis_overwrite,
                    vis_anim=args.vis_anim,
                    num_rep=i,
                    export_results=args.vis_export,
                )
        if args.vis_gt:
            logger.info("Visualizing the ground truth...")
            visualizer.visualize_all(
                output_dir, gt_data=True, overwrite=args.vis_overwrite
            )
            """
            visualizer.visualize_subset(
                output_dir,
                gt_data=True,
                overwrite=args.vis_overwrite,
                export_results=args.vis_export,
            )
            """

    if args.eval:
        logger.info("Evaluating the model...")
        evaluator = EvaluatorWrapper(
            args,
            generator,
            dataset,
            body_model,
            device,
            batch_size=args.eval_batch_size,
        )
        log, all_results_df, arr_based_metrics = evaluator.evaluate_all()
        csv_path = output_dir / f"results_{args.dataset}.csv"
        evaluator.store_all_results(all_results_df, csv_path)
        evaluator.store_plots(arr_based_metrics, output_dir)
        evaluator.print_results(log)


if __name__ == "__main__":
    main()
