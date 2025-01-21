# MIT License
# Copyright (c) 2022 Guy Tevet
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from diffusion import gaussian_diffusion as gd
from diffusion.respace import space_timesteps, SpacedDiffusion, SpacedRollingDiffusion
from loguru import logger
from model.maskers import create_masker
from model.mdm_model import RollingMDM, RollingTransformer, StandardMDM
from model.meta_model import MetaModel, MetaModelRolling
from model.model_wrapper import ModelWrapper
from utils.constants import DiffusionType


def load_diffusion_model(args, device="cuda"):
    logger.info("Creating model and diffusion...")
    model_cls = get_model_class(args)
    model, diffusion = create_model_and_diffusion(
        args,
        model_cls=model_cls,
    )

    logger.info(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location=device)
    load_model_wo_clip(model, state_dict)

    model.to(device)  # dist_util.dev())
    model.eval()  # disable random masking
    return model, diffusion


def get_model_class(args):
    diffusion_type, arch = args.diffusion_type, args.arch
    if arch == "diffusion_DiffMLP":
        arch = "DiffMLP"  # retrocompatibility
        args.arch = "DiffMLP"

    # ======= Standard diffusion ========
    if diffusion_type == DiffusionType.STANDARD:
        if arch == "DiffMLP":
            return MetaModel
        elif arch == "MDM":
            return StandardMDM
        else:
            raise ValueError("Only DiffMLP is supported for standard diffusion")

    # ======= Rolling diffusion ========
    if arch == "DiffMLP":
        return MetaModelRolling
    elif arch == "MDM":
        return RollingMDM
    elif arch == "Transformer":
        return RollingTransformer
    else:
        raise NotImplementedError


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


def create_model_and_diffusion(args, model_cls=MetaModel):
    model = model_cls(**get_model_args(args, model_cls))
    diffusion = create_gaussian_diffusion(args)
    model = ModelWrapper(model, diffusion, args.target_type, args.prediction_input_type)
    return model, diffusion


def get_model_args(args, model_cls):
    mask_cond_fn = create_masker(
        args.masker,
        args.dataset,
        args.cond_mask_prob,
        min_f=args.masker_minf,
        max_f=args.masker_maxf,
    )
    args_dict = {
        "arch": args.arch,
        "nfeats": args.motion_nfeat,
        "mask_cond_fn": mask_cond_fn,
        "latent_dim": args.latent_dim,
        "sparse_dim": args.sparse_dim,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "dataset": args.dataset,
        "input_motion_length": args.input_motion_length,
    }

    if model_cls == MetaModelRolling:
        args_dict.update(
            {
                "rolling_motion_ctx": args.rolling_motion_ctx,
                "rolling_sparse_ctx": args.rolling_sparse_ctx,
                "framewise_time_emb": args.framewise_time_emb,
            }
        )
    elif model_cls in [RollingTransformer, RollingMDM]:
        args_dict.update(
            {
                "rolling_motion_ctx": args.rolling_motion_ctx,
                "rolling_sparse_ctx": args.rolling_sparse_ctx,
                "ff_size": args.ff_size,
                "num_heads": args.num_heads,
                "activation": args.activation,
                "dropout_framewise": args.dropout_framewise,
                "mdm_timestep_emb": args.mdm_timestep_emb,
                "lookahead": args.lookahead,
                "use_shape_head": args.use_shape_head,
            }
        )
        if model_cls == RollingMDM:
            args_dict.update(
                {
                    "mdm_timestep_emb": args.mdm_timestep_emb,
                    "cond": args.cond,
                }
            )
        elif model_cls == RollingTransformer:
            args_dict.update(
                {
                    "time_emb_strategy": args.time_emb_strategy,
                    "num_layers_cond": args.layers_cond,
                }
            )
    if args.diffusion_type == DiffusionType.ROLLING:
        args_dict.update(
            {
                "mask_cond_fn": create_masker(
                    args.masker, args.dataset, 0.0
                )  # masker is in diffusion class for rollingDM
            }
        )
    return args_dict


def create_gaussian_diffusion(args):
    predict_xstart = True
    steps = args.diffusion_steps  # 1000
    scale_beta = 1.0
    timestep_respacing = args.timestep_respacing
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    if args.diffusion_type == DiffusionType.ROLLING:
        mask_cond_fn = create_masker(
            args.masker,
            args.dataset,
            args.cond_mask_prob,
            min_f=args.masker_minf,
            max_f=args.masker_maxf,
        )
        return SpacedRollingDiffusion(
            dataset=args.dataset,
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON
                if not predict_xstart
                else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not args.sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            rolling_type=args.rolling_type,
            rolling_motion_ctx=args.rolling_motion_ctx,
            rolling_sparse_ctx=args.rolling_sparse_ctx,
            rolling_fr_frames=args.rolling_fr_frames,
            rolling_fr_bptt=args.rolling_fr_bptt,
            rolling_fr_jump=args.rolling_fr_jump,
            target_type=args.target_type,
            clamp_noise=args.clamp_noise,
            ctx_perturbation=args.ctx_perturbation,
            sp_perturbation=args.sp_perturbation,
            mse_loss_weight_type=args.loss_weights,
            mask_cond_fn=mask_cond_fn,
            loss_dist_type=args.loss_dist_type,
            loss_velocity=args.loss_velocity,
            loss_jitter=args.loss_jitter,
            loss_correction=args.loss_correction,
            loss_fk=args.loss_fk,
            loss_fk_vel=args.loss_fk_vel,
            loss_fk_vel_avg=args.loss_fk_vel_avg,
            loss_fk_vel_feet=args.loss_fk_vel_feet,
            support_dir=args.support_dir,
        )
    elif args.diffusion_type == DiffusionType.STANDARD:
        return SpacedDiffusion(
            dataset=args.dataset,
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON
                if not predict_xstart
                else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not args.sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            mse_loss_weight_type=args.loss_weights,
            loss_dist_type=args.loss_dist_type,
            loss_velocity=args.loss_velocity,
            loss_jitter=args.loss_jitter,
            loss_fk=args.loss_fk,
            support_dir=args.support_dir,
        )
    else:
        raise ValueError(f"Unknown diffusion type {args.diffusion_type}")
