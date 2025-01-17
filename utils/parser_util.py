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

from utils.config import pathmgr
from utils.constants import (
    BackpropThroughTimeType,
    ConditionMasker,
    DatasetType,
    DiffusionType,
    FreeRunningJumpType,
    LossDistType,
    NoiseScheduleType,
    PredictionInputType,
    PredictionTargetType,
    RollingType,
    RollingVisType,
)


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    if args.dataset == DatasetType.DEFAULT:
        #  if DEFAULT --> Different dataset is not specified by CMD --> use the one from the model
        args_to_overwrite = ["support_dir", "dataset", "dataset_path", "results_dir"]
    for group_name in [
        "model",
        "diffusion",
    ]:  # "dataset"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = pathmgr.get_local_path(
        os.path.join(os.path.dirname(model_path), "args.json")
    )
    assert os.path.exists(args_path), "Arguments json file was not found!"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)
    for a in args_to_overwrite:
        if a in model_args.keys():
            args.__dict__[a] = model_args[a]
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
    return ValueError("group_name was not found.")


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
        "--batch_size", default=64, type=int, help="Batch size during training."
    )
    group.add_argument(
        "--timestep_respacing", default="", type=str, help="ddim timestep respacing."
    )
    group.add_argument(
        "--results_dir",
        default="manifold://xr_body/tree/personal/gbarquero/omp",
        # required=True,
        type=str,
        help="Path to save checkpoints, logs, and results.",
    )


def add_diffusion_options(parser):
    group = parser.add_argument_group("diffusion")
    group.add_argument(
        "--noise_schedule",
        default=NoiseScheduleType.COSINE,
        type=NoiseScheduleType,
        help="Noise schedule type",
    )
    group.add_argument(
        "--diffusion_steps",
        default=1000,
        type=int,
        help="Number of diffusion steps (denoted T in the paper)",
    )
    group.add_argument(
        "--sigma_small", default=True, type=bool, help="Use smaller sigma values."
    )
    group.add_argument(
        "--loss_weights",
        default="constant",
        type=str,
        help="Weighting of the loss. Can be constant or time-based.",
    )
    group.add_argument(
        "--loss_dist_type",
        default=LossDistType.L2,
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
        "--loss_jitter",
        default=0.0,
        type=float,
        help="Weighting of the prediction jitter loss.",
    )
    group.add_argument(
        "--loss_correction",
        default=0.0,
        type=float,
        help="Weighting of the correction loss.",
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
    group.add_argument(
        "--loss_fk_vel_avg",
        default=0.0,
        type=float,
        help="Weighting of the loss applied to the average velocity in the whole prediction window.",
    )
    group.add_argument(
        "--loss_fk_vel_feet",
        default=0.0,
        type=float,
        help="Weighting of the loss applied to the average velocity in the whole prediction window.",
    )


def add_model_options(parser):
    group = parser.add_argument_group("model")
    group.add_argument(
        "--arch",
        choices=["DiffMLP", "MDM", "Transformer"],
        default="DiffMLP",
        type=str,
        help="Architecture types as reported in the paper.",
    )
    group.add_argument(
        "--diffusion_type",
        default=DiffusionType.STANDARD,
        type=DiffusionType,
        help="Diffusion type as reported in the paper.",
    )
    group.add_argument(
        "--masker",
        default=ConditionMasker.SEQ_ALL,
        type=ConditionMasker,
        help="Type of masker for unconditional generation, or CFG (e.g., default, independent)",
    )
    group.add_argument(
        "--masker_minf",
        default=0,
        type=int,
        help="Min frames of the masked segment (for segment-wise maskers only)",
    )
    group.add_argument(
        "--masker_maxf",
        default=0,
        type=int,
        help="Max frames of the masked segment (for segment-wise maskers only)",
    )
    group.add_argument(
        "--target_type",
        default=PredictionTargetType.POSITIONS,
        type=PredictionTargetType,
        help="target type (position, relative offset, absolute offset)",
    )
    group.add_argument(
        "--prediction_input_type",
        default=PredictionInputType.NOISY,
        type=PredictionInputType,
        help="type of previous prediction fed into the network (noisy, clean, none)",
    )
    group.add_argument(
        "--clamp_noise",
        action="store_true",
        help="If noise is clamped to [-1, 1] when q-sampling",
    )
    group.add_argument(
        "--rolling_type",
        default=RollingType.ROLLING,
        type=RollingType,
        help="Rolling diffusion type (rolling, omp, diffusionforcing)",
    )
    group.add_argument(
        "--rolling_context",
        default=-1,
        type=int,
        help="[DEPRECATED] Uses the same length of context for both tracking signal and motion contexts",
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
        "--rolling_fr_bptt",
        default=BackpropThroughTimeType.NONE,
        type=BackpropThroughTimeType,
        help="type of backprop through time",
    )
    group.add_argument(
        "--rolling_fr_jump",
        default=FreeRunningJumpType.NONE,
        type=FreeRunningJumpType,
        help="jump randomly during the Free Running process",
    )
    group.add_argument(
        "--rolling_latency",
        default=0,
        type=int,
        help="num of frames of latency (access to sparse info in the future)",
    )
    group.add_argument(
        "--ctx_perturbation",
        default=0,
        type=float,
        help="noise std for context perturbation (to avoid degeneration at inference)",
    )
    group.add_argument(
        "--sp_perturbation",
        default=0,
        type=float,
        help="max noise std for random sparse perturbation",
    )
    group.add_argument(
        "--motion_nfeat", default=132, type=int, help="motion feature dimension"
    )
    group.add_argument(
        "--sparse_dim", default=54, type=int, help="sparse signal feature dimension"
    )
    group.add_argument("--layers", default=8, type=int, help="Number of layers.")
    group.add_argument(
        "--latent_dim", default=512, type=int, help="Transformer/GRU width."
    )
    group.add_argument(
        "--cond_mask_prob",
        default=0.0,
        type=float,
        help="The probability of masking the condition during training."
        " For classifier-free guidance learning.",
    )
    group.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="The dropout probability.",
    )
    group.add_argument(
        "--input_motion_length",
        default=196,
        type=int,
        help="Limit for the maximal number of frames.",
    )
    group.add_argument(
        "--no_normalization",
        action="store_true",
        help="no data normalisation for the 6d motions",
    )
    group.add_argument(
        "--framewise_time_emb",
        action="store_true",
        help="frame-wise time embedding (for rolling diffusion)",
    )
    # ======== MDM PARAMS ========
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
    group.add_argument(
        "--dropout_framewise",
        default=0.0,
        type=float,
        help="The dropout probability for each frame in the sequence to denoise.",
    )
    group.add_argument(
        "--mdm_timestep_emb",
        action="store_true",
        help="[using MDM arch] use a seq-wise time embedding for each frame in the sequence to denoise.",
    )
    group.add_argument(
        "--lookahead",
        action="store_true",
        help="[using MDM arch] lookahead attention (the inverted version of causal attention)",
    )
    group.add_argument(
        "--cond",
        type=str,
        help="[using MDM arch] type of conditioning for sparse signal (tracking)",
        default="concat",
        choices=["concat", "xatt"],
    )
    # ======== Transformer PARAMS ========
    group.add_argument(
        "--layers_cond",
        default=1,
        type=int,
        help="[with Transformer] Number of layers for the encoder of the condition (ctx + sparse signal).",
    )
    group.add_argument(
        "--time_emb_strategy",
        default="concat",
        help="[with Transformer] type of diffusion timestep embedding strategy (e.g., concat, add, norm).",
    )
    group.add_argument(
        "--use_shape_head",
        action="store_true",
        help="activate the shape head (for blending shape regression)",
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
        default="manifold://xr_body/tree/personal/gbarquero/datasets/agrol/AMASS/new_format_data",
        type=str,
        help="Dataset path",
    )
    group.add_argument(
        "--support_dir",
        type=str,
        default="manifold://xr_body/tree/personal/gbarquero/datasets/agrol/SMPL/",
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
    group.add_argument("--lr", default=2e-4, type=float, help="Learning rate.")
    group.add_argument(
        "--weight_decay", default=0.0, type=float, help="Optimizer weight decay."
    )
    group.add_argument(
        "--lr_anneal_steps",
        default=0,
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
        default=5000,
        type=int,
        help="Save checkpoints and run evaluation each N steps",
    )
    group.add_argument(
        "--num_steps",
        default=6000000,
        type=int,
        help="Training will stop after the specified number of steps.",
    )
    group.add_argument(
        "--resume_checkpoint",
        default="",
        type=str,
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
        "--random",
        action="store_true",
        help="random order of the evaluation dataset",
    )
    group.add_argument(
        "--overlapping_test",
        action="store_true",
        help="enabling overlapping test",
    )
    group.add_argument(
        "--uncond",
        action="store_true",
        help="unconditional sampling",
    )
    group.add_argument(
        "--num_per_batch",
        default=256,
        type=int,
        help="the batch size of each split during non-overlapping testing",
    )
    group.add_argument(
        "--sld_wind_size",
        default=70,
        type=int,
        help="the sliding window size",
    )
    group.add_argument(
        "--eval",
        action="store_true",
        help="evaluate the model",
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
        "--vis_anim",
        default=RollingVisType.NONE,
        type=RollingVisType,
        help="type of rolling animation to visualize",
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
    group.add_argument(
        "--vis_reps",
        default=1,
        type=int,
        help="Number of repetitions of the visualization.",
    )
    group.add_argument(
        "--animation",
        action="store_true",
        help="visualize the rolling animation",
    )
    group.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument(
        "--rolling_horizon",
        default=-1,
        type=int,
        help="the horizon of the rolling noise schedule. If -1, will use the default training value. It has to be < input_motion_length",
    )
    group.add_argument(
        "--init_ik",
        action="store_true",
        help="if true, will initialize the rolling diffusion with T-pose (oriented according to headset) + IK on the controllers/tracking wrist points",
    )
    group.add_argument(
        "--cfg",
        default=1.0,
        type=float,
        help="Classifier-free guidance value (default=1.0)",
    )
    group.add_argument(
        "--cfg_min_snr",
        default=-float("inf"),
        type=float,
        help="CFG in intervals (https://arxiv.org/pdf/2404.07724) - min snr where applied",
    )
    group.add_argument(
        "--cfg_max_snr",
        default=float("inf"),
        type=float,
        help="CFG in intervals (https://arxiv.org/pdf/2404.07724) - max snr where applied",
    )
    group.add_argument(
        "--eval_batch_size", default=1, type=int, help="Batch size during sampling."
    )
    group.add_argument(
        "--precomputed_path",
        default=None,
        type=str,
        help="Path to precomputed data (e.g., for visualization of baselines).",
    )


def add_evaluation_options(parser):
    group = parser.add_argument_group("eval")
    group.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
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
