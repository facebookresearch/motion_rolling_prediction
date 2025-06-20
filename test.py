# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch

from data_loaders.dataloader import TestDataset
from evaluation.evaluation import EvaluatorWrapper
from evaluation.generators import create_generator

from evaluation.utils import BodyModelsWrapper
from evaluation.visualization import VisualizerWrapper

from loguru import logger

from utils.model_util import load_rpm_model
from utils.parser_util import sample_args


def main():
    args = sample_args()
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

    generator = create_generator(args, model, dataset, device)
    suffix = generator.get_folder_suffix()
    subfolder_name += suffix
    name_results_folder = "results"
    if args.test_split != "test":
        name_results_folder += f"_{args.test_split}"
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
            # visualize the test subset of the dataset
            visualizer.visualize_subset(
                output_dir,
                overwrite=args.vis_overwrite,
                export_results=args.vis_export,
            )
        if args.vis_gt:
            logger.info("Visualizing the ground truth...")
            # visualizer.visualize_all(
            #     output_dir, gt_data=True, overwrite=args.vis_overwrite
            # )
            visualizer.visualize_subset(
                output_dir,
                gt_data=True,
                overwrite=args.vis_overwrite,
                export_results=args.vis_export,
            )

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
        more_metrics = evaluator.store_plots(arr_based_metrics, output_dir)
        log.update(more_metrics)
        evaluator.print_results(log)


if __name__ == "__main__":
    main()
