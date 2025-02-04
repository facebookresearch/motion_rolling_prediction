# MIT License
# Copyright (c) 2022 Guy Tevet
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

from utils.constants import (
    ConditionMasker,
    DatasetType,
    LossDistType,
    PredictionInputType,
    PredictionTargetType,
)


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_losses_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    if args.dataset == DatasetType.DEFAULT:
        #  if DEFAULT --> Different dataset is not specified by CMD --> use the one from the model
        args_to_overwrite = ["support_dir", "dataset", "dataset_path", "results_dir"]
    for group_name in [
        "model",
        "losses",
    ]:  # "dataset"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    assert os.path.exists(args_path), "Arguments json file was not found!"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)
    for a in args_to_overwrite:
        if a in model_args.keys():
            args.__dict__[a] = model_args[a]
            if a == "dataset":
                args.__dict__[a] = DatasetType(model_args[a])
            elif "_dir" in a or "_path" in a:
                args.__dict__[a] = Path(model_args[a])
        else:
            print(
                "Warning: was not able to load [{}], using default value [{}] instead.".format(
                    a, args.__dict__[a]
                )
            )
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    raise ValueError("group_name was not found.")


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("model_path")
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except Exception:
        raise ValueError("model_path argument must be specified.")


def add_base_options(parser):
    group = parser.add_argument_group("base")
    group.add_argument(
        "--cpu",
        action="store_true",
        help="If True, will use cpu, if not cuda.",
    )
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument(
        "--batch_size", default=32, type=int, help="Batch size during training."
    )
    group.add_argument(
        "--results_dir",
        default="./results",
        type=Path,
        help="Path to save checkpoints, logs, and results.",
    )


def add_losses_options(parser):
    group = parser.add_argument_group("losses")
    group.add_argument(
        "--loss_dist_type",
        default=LossDistType.L1,
        type=LossDistType,
        help="Type of loss to use (L1, L2).",
    )
    group.add_argument(
        "--loss_velocity",
        default=0.0,
        type=float,
        help="Weighting of the velocity loss.",
    )
    group.add_argument(
        "--loss_fk",
        default=0.0,
        type=float,
        help="Weighting of the loss applied to the joints after running the differentiable FK.",
    )
    group.add_argument(
        "--loss_fk_vel",
        default=0.0,
        type=float,
        help="Weighting of the loss applied to the velocity of the joints.",
    )


def add_model_options(parser):
    group = parser.add_argument_group("model")
    group.add_argument(
        "--masker",
        default=ConditionMasker.SEG_HANDS_IDP,
        type=ConditionMasker,
        help="Type of masker for unconditional generation (e.g., default, independent)",
    )
    group.add_argument(
        "--masker_minf",
        default=0,
        type=int,
        help="Min frames of the masked segment (for segment-wise maskers only)",
    )
    group.add_argument(
        "--masker_maxf",
        default=sys.maxsize,
        type=int,
        help="Max frames of the masked segment (for segment-wise maskers only)",
    )
    group.add_argument(
        "--target_type",
        default=PredictionTargetType.PCAF_COSINE,
        type=PredictionTargetType,
        help="target type (position, pcaf_cosine, pcaf_cosinesq, pcaf_linear)",
    )
    group.add_argument(
        "--prediction_input_type",
        default=PredictionInputType.NONE,
        type=PredictionInputType,
        help="type of previous prediction fed into the network (noisy, clean, none)",
    )
    group.add_argument(
        "--rolling_motion_ctx",
        default=0,
        type=int,
        help="num of past motion frames provided as context",
    )
    group.add_argument(
        "--rolling_sparse_ctx",
        default=0,
        type=int,
        help="num of past frames for which tracking signal is provided as context (plus present)",
    )
    group.add_argument(
        "--rolling_fr_frames",
        default=0,
        type=int,
        help="num of max iterations of free running to be done",
    )
    group.add_argument(
        "--rolling_latency",
        default=0,
        type=int,
        help="num of frames of latency (access to sparse info in the future)",
    )
    group.add_argument(
        "--motion_nfeat", default=132, type=int, help="motion feature dimension"
    )
    group.add_argument(
        "--sparse_dim", default=54, type=int, help="sparse signal feature dimension"
    )
    group.add_argument("--layers", default=4, type=int, help="Number of layers.")
    group.add_argument(
        "--latent_dim", default=512, type=int, help="Transformer/GRU width."
    )
    group.add_argument(
        "--cond_mask_prob",
        default=0.1,
        type=float,
        help="The probability of masking the condition during training.",
    )
    group.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="The dropout probability.",
    )
    group.add_argument(
        "--input_motion_length",
        default=10,
        type=int,
        help="Limit for the maximal number of frames.",
    )
    group.add_argument(
        "--no_normalization",
        action="store_true",
        help="no data normalisation for the 6d motions",
    )
    group.add_argument(
        "--ff_size",
        default=1024,
        type=int,
        help="size of feedforward layers",
    )
    group.add_argument(
        "--num_heads",
        default=4,
        type=int,
        help="num of heads in the attention layers",
    )
    group.add_argument(
        "--activation",
        default="gelu",
        type=str,
        help="activation layer",
    )


def add_data_options(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--dataset",
        default=DatasetType.DEFAULT,
        type=DatasetType,
        help="Dataset name.",
    )
    group.add_argument(
        "--dataset_path",
        default="./datasets_processed/",
        type=Path,
        help="Dataset path",
    )
    group.add_argument(
        "--support_dir",
        type=Path,
        default="./SMPL/",
        help="the dir that you store your smplh and dmpls dirs",
    )
    group.add_argument(
        "--dataset_max_samples",
        default=-1,
        type=int,
        help="Num of dataset sequences to consider",
    )
    group.add_argument(
        "--min_frames",
        default=0,
        type=int,
        help="Lower bound of the number of frames. Below, they are skipped.",
    )
    group.add_argument(
        "--max_frames",
        default=sys.maxsize,
        type=int,
        help="Upper bound of the number of frames. Above, they are skipped.",
    )
    group.add_argument(
        "--eval_gap_config",
        type=str,
        default=None,
        help="Config file with the gaps configuration for evaluating the tracking signal loss.",
    )
    group.add_argument(
        "--use_real_input",
        action="store_true",
        help="use real tracking input signal",
    )
    group.add_argument(
        "--input_conf_threshold",
        type=float,
        default=0.0,
        help="confidence threshold when using real tracking input signal",
    )
    group.add_argument(
        "--test_split",
        type=str,
        default="test",
        help="test split to use: test, test_controllers, test_tracking (these two only for GORP)",
    )


def add_training_options(parser):
    group = parser.add_argument_group("training")
    group.add_argument(
        "--exp_name",
        required=True,
        type=str,
        help="Name of the experiment",
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="If True, will enable to use an already existing save_dir.",
    )
    group.add_argument(
        "--train_platform_type",
        default="NoPlatform",
        choices=["NoPlatform", "ClearmlPlatform", "TensorboardPlatform"],
        type=str,
        help="Choose platform to log results. NoPlatform means no logging.",
    )
    group.add_argument("--lr", default=3e-4, type=float, help="Learning rate.")
    group.add_argument(
        "--weight_decay", default=1e-4, type=float, help="Optimizer weight decay."
    )
    group.add_argument(
        "--lr_anneal_steps",
        default=50000,
        type=int,
        help="Number of learning rate anneal steps.",
    )
    group.add_argument(
        "--train_dataset_repeat_times",
        default=1000,
        type=int,
        help="Repeat the training dataset to save training time",
    )
    group.add_argument(
        "--eval_during_training",
        action="store_true",
        help="If True, will run evaluation during training.",
    )
    group.add_argument(
        "--vis_during_training",
        action="store_true",
        help="If True, will run visualization during training.",
    )
    group.add_argument(
        "--log_interval", default=1, type=int, help="Log losses each N steps"
    )
    group.add_argument(
        "--save_interval",
        default=10,
        type=int,
        help="Save checkpoints and run evaluation each N steps",
    )
    group.add_argument(
        "--num_steps",
        default=100000,
        type=int,
        help="Training will stop after the specified number of steps.",
    )
    group.add_argument(
        "--resume_checkpoint",
        default=None,
        type=Path,
        help="If not empty, will start from the specified checkpoint (path to model###.pt file).",
    )
    group.add_argument(
        "--load_optimizer",
        action="store_true",
        help="If True, will also load the saved optimizer state for network initialization",
    )
    group.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of dataloader workers.",
    )


def add_sampling_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--model_path",
        required=True,
        type=Path,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument(
        "--random",
        action="store_true",
        help="random order of the evaluation dataset",
    )
    group.add_argument(
        "--eval",
        action="store_true",
        help="evaluate the model",
    )
    group.add_argument(
        "--eval_batch_size", default=1, type=int, help="Batch size during sampling."
    )
    group.add_argument(
        "--vis",
        action="store_true",
        help="visualize the output",
    )
    group.add_argument(
        "--vis_gt",
        action="store_true",
        help="visualize the GT",
    )
    group.add_argument(
        "--vis_overwrite",
        action="store_true",
        help="If True, will enable to overwrite previously generated visualizations.",
    )
    group.add_argument(
        "--vis_export",
        action="store_true",
        help="If True, it will export all data needed for building the Unity scene.",
    )


def add_evaluation_options(parser):
    group = parser.add_argument_group("eval")
    group.add_argument(
        "--model_path",
        required=True,
        type=Path,
        help="Path to model####.pt file to be sampled.",
    )


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_losses_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def sample_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)
