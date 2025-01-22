# MIT License
# Copyright (c) 2022 Guy Tevet
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from loguru import logger
from rolling.rolling_model import RollingPredictionModel
from model.maskers import create_masker
from model.mdm_model import RollingMDM
from model.model_wrapper import ModelWrapper


def load_rpm_model(args, device="cuda"):
    logger.info("Creating model and rpm...")
    model, rpm = create_model_and_rpm(args)

    logger.info(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location=device)
    load_model_wo_clip(model, state_dict)

    model.to(device)  # dist_util.dev())
    model.eval()  # disable random masking
    return model, rpm


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(unexpected_keys) != 0:
        state_dict_new = {}
        for key in state_dict.keys():
            state_dict_new[key.replace("module.", "")] = state_dict[key]
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict_new, strict=False
        )
    logger.warning(f"unexpected keys: {unexpected_keys}")
    # assert len(unexpected_keys) == 0, unexpected_keys
    assert all([k.startswith("clip_model.") for k in missing_keys])


def create_model_and_rpm(args):
    model = RollingMDM(**get_model_args(args))
    rpm = create_gaussian_rpm(args)
    model = ModelWrapper(model, args.target_type, args.prediction_input_type, args.input_motion_length)
    return model, rpm


def get_model_args(args):
    return {
        "nfeats": args.motion_nfeat,
        "latent_dim": args.latent_dim,
        "sparse_dim": args.sparse_dim,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "dataset": args.dataset,
        "input_motion_length": args.input_motion_length,
        "rolling_motion_ctx": args.rolling_motion_ctx,
        "rolling_sparse_ctx": args.rolling_sparse_ctx,
        "ff_size": args.ff_size,
        "num_heads": args.num_heads,
        "activation": args.activation,
        "dropout_framewise": args.dropout_framewise,
        "mdm_timestep_emb": args.mdm_timestep_emb,
        "lookahead": args.lookahead,
        "use_shape_head": args.use_shape_head,
        "mdm_timestep_emb": args.mdm_timestep_emb,
        "cond": args.cond,
    }


def create_gaussian_rpm(args):
    mask_cond_fn = create_masker(
        args.masker,
        args.dataset,
        args.cond_mask_prob,
        min_f=args.masker_minf,
        max_f=args.masker_maxf,
    )
    return RollingPredictionModel(
        rolling_type=args.rolling_type,
        rolling_motion_ctx=args.rolling_motion_ctx,
        rolling_sparse_ctx=args.rolling_sparse_ctx,
        rolling_fr_frames=args.rolling_fr_frames,
        target_type=args.target_type,
        ctx_perturbation=args.ctx_perturbation,
        sp_perturbation=args.sp_perturbation,
        mask_cond_fn=mask_cond_fn,
        loss_dist_type=args.loss_dist_type,
        loss_velocity=args.loss_velocity,
        loss_fk=args.loss_fk,
        loss_fk_vel=args.loss_fk_vel,
        support_dir=args.support_dir,
    )
