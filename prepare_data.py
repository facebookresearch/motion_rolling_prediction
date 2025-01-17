# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import os
import tempfile

import numpy as np
import torch

from evaluation.utils import BodyModelsWrapper
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from loguru import logger
from tqdm import tqdm
from utils import utils_transform
from utils.config import pathmgr
from utils.constants import SMPLGenderParam, SMPLModelType


def from_smpl_to_input_features(
    smpl_pose_vec: torch.Tensor, pose_joints_world: torch.Tensor, kintree
):
    """
    smpl_pose_vec: [num_frames, 66] -> pose of the body in SMPL format
    pose_joints: [num_frames, 22, 3] -> position of the joints in the world coordinate system
    """
    gt_rotations_aa = torch.Tensor(smpl_pose_vec[:, :66]).reshape(-1, 3)
    gt_rotations_6d = utils_transform.aa2sixd(gt_rotations_aa).reshape(
        smpl_pose_vec.shape[0], -1
    )

    rotation_local_matrot = aa2matrot(
        torch.tensor(smpl_pose_vec).reshape(-1, 3)
    ).reshape(smpl_pose_vec.shape[0], -1, 9)
    rotation_global_matrot = local2global_pose(
        rotation_local_matrot, kintree[0].long()
    )  # rotation of joints relative to the origin
    head_rotation_global_matrot = rotation_global_matrot[1:, 15, :, :]

    rotation_global_6d = utils_transform.matrot2sixd(
        rotation_global_matrot.reshape(-1, 3, 3)
    ).reshape(rotation_global_matrot.shape[0], -1, 6)
    input_rotation_global_6d = rotation_global_6d[1:, [15, 20, 21], :]

    rotation_velocity_global_matrot = torch.matmul(
        torch.inverse(rotation_global_matrot[:-1]),
        rotation_global_matrot[1:],
    )
    rotation_velocity_global_6d = utils_transform.matrot2sixd(
        rotation_velocity_global_matrot.reshape(-1, 3, 3)
    ).reshape(rotation_velocity_global_matrot.shape[0], -1, 6)
    input_rotation_velocity_global_6d = rotation_velocity_global_6d[:, [15, 20, 21], :]

    num_frames = pose_joints_world.shape[0] - 1
    hmd_cond = torch.cat(
        [
            input_rotation_global_6d.reshape(num_frames, -1),
            input_rotation_velocity_global_6d.reshape(num_frames, -1),
            pose_joints_world[1:, [15, 20, 21], :].reshape(num_frames, -1),
            pose_joints_world[1:, [15, 20, 21], :].reshape(num_frames, -1)
            - pose_joints_world[:-1, [15, 20, 21], :].reshape(num_frames, -1),
        ],
        dim=-1,
    )

    position_head_world = pose_joints_world[1:, 15, :]  # world position of head
    head_global_trans = torch.eye(4).repeat(num_frames, 1, 1)
    head_global_trans[:, :3, :3] = head_rotation_global_matrot
    head_global_trans[:, :3, 3] = position_head_world

    data = {
        "rotation_local_full_gt_list": gt_rotations_6d[1:],
        "rotation_global_full_gt_list": rotation_global_6d[1:, :22]
        .reshape(num_frames, -1)
        .cpu()
        .float(),
        "hmd_position_global_full_gt_list": hmd_cond,
        "head_global_trans_list": head_global_trans,
        "position_global_full_gt_world": pose_joints_world[1:].cpu().float(),
    }
    return data


def main(args, device="cuda:0"):
    if args.use_vip:
        from iopath import PathManager
        from iopath.fb.manifold import ManifoldPathHandler

        vip_pathmgr = PathManager()
        vip_pathmgr.register_handler(ManifoldPathHandler(use_vip=True))
    else:
        vip_pathmgr = pathmgr

    # We use male/female as specified in AMASS metadata for each sequence
    # It will be initalized after reading the first sequence (smplh or smplx)
    body_model = None
    body_model_type = ""

    # list folders of args.splits_dir
    all_datasets = sorted(os.listdir(args.splits_dir))
    for dataroot_subset in all_datasets:
        for phase in ["train", "test"]:
            logger.info(f"Processing {dataroot_subset} {phase}...")
            split_file = os.path.join(
                args.splits_dir, dataroot_subset, phase + "_split.txt"
            )
            if not pathmgr.exists(split_file):
                logger.info(f"{split_file} does not exist, skipping...")
                continue

            savedir = os.path.join(args.save_dir, dataroot_subset, phase)
            pathmgr.mkdirs(savedir)

            with open(split_file, "r") as f:
                filepaths = [line.strip() for line in f]

            idx = 0
            pbar = tqdm(total=len(filepaths))
            while idx < len(filepaths):
                filepath = filepaths[idx]
                idx += 1
                dst_fname = "{}.pt".format(idx)
                manifold_path = os.path.join(savedir, dst_fname)
                if pathmgr.exists(manifold_path):
                    pbar.update(1)
                    continue

                bdata = np.load(
                    vip_pathmgr.get_local_path(os.path.join(args.root_dir, filepath)),
                    allow_pickle=True,
                )

                if "mocap_framerate" in bdata:
                    fps = bdata["mocap_framerate"]
                elif "mocap_frame_rate" in bdata:
                    fps = bdata["mocap_frame_rate"]
                else:
                    logger.info("No mocap_framerate found in {}".format(filepath))
                    continue

                # type of body model
                if "surface_model_type" in bdata:
                    new_body_model_type = SMPLModelType.parse(
                        bdata["surface_model_type"].item()
                    )
                    assert (
                        body_model_type == "" or body_model_type == new_body_model_type
                    ), "Can't mix different body models: {} vs {}".format(
                        body_model_type, new_body_model_type
                    )
                    body_model_type = SMPLModelType.parse(
                        bdata["surface_model_type"].item()
                    )
                else:
                    body_model_type = SMPLModelType.SMPLH  # by default

                if body_model is None:
                    logger.info("Initializing body model: {}".format(body_model_type))
                    body_model = BodyModelsWrapper(args.support_dir)

                num_frames = bdata["trans"].shape[0]

                if args.out_fps > fps:
                    raise AssertionError("Cannot supersample data!")
                else:
                    fps_ratio = float(args.out_fps) / fps
                    new_num_frames = int(fps_ratio * num_frames)
                    # to keep retrocompatibility with AGRoL data preprocessing.
                    last_frame = (
                        num_frames - 2 if num_frames % 2 == 0 else num_frames - 1
                    )
                    downsamp_inds = np.linspace(
                        0, last_frame, num=new_num_frames, dtype=int
                    )

                    # update data to save
                    fps = args.out_fps
                    num_frames = new_num_frames

                if num_frames == 0:
                    logger.info("No frames found in {}".format(filepath))
                    continue
                bdata_poses = bdata["poses"][downsamp_inds, ...]
                bdata_trans = bdata["trans"][downsamp_inds, ...]
                smpl_gender = (
                    "neutral" if args.use_neutral_shape else str(bdata["gender"])
                )

                body_parms = {
                    "root_orient": torch.Tensor(
                        bdata_poses[:, :3]
                    ),  # controls the global root orientation
                    "pose_body": torch.Tensor(
                        bdata_poses[:, 3:66]
                    ),  # controls the body
                    "trans": torch.Tensor(
                        bdata_trans
                    ),  # controls the global body position
                }
                if not args.use_neutral_shape:
                    bdata_betas = bdata["betas"][:16]
                    body_parms["betas"] = torch.Tensor(bdata_betas).repeat(
                        bdata_poses.shape[0], 1
                    )
                body_pose_world = body_model(
                    {k: v.to(device) for k, v in body_parms.items()},
                    body_model_type,
                    SMPLGenderParam[smpl_gender.upper()],
                )
                gt_joints_world_space = body_pose_world.Jtr[
                    :, :22, :
                ].cpu()  # position of joints relative to the world origin

                kintree = body_model.get_kin_tree(
                    body_model_type, SMPLGenderParam[smpl_gender.upper()]
                )
                data = from_smpl_to_input_features(
                    bdata_poses,
                    gt_joints_world_space,
                    kintree,
                )
                data["body_parms_list"] = body_parms
                data["framerate"] = fps
                data["gender"] = smpl_gender
                data["filepath"] = filepath
                data["surface_model_type"] = body_model_type

                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmp_local_path = os.path.join(tmpdirname, dst_fname)
                    torch.save(data, tmp_local_path)
                    pathmgr.copy(
                        src_path=tmp_local_path, dst_path=manifold_path, overwrite=True
                    )
                pbar.update(1)


def run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="prepare_data/data_split",
        help="=dir where the data splits are defined",
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default=None,
        help="=dir where you put your smplh and dmpls dirs",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="=dir where you want to save your generated data",
    )
    parser.add_argument(
        "--root_dir", type=str, default=None, help="=dir where you put your AMASS data"
    )
    parser.add_argument(
        "--use_vip",
        action="store_true",
        help="If True, it will use the --vip argument when accessing the Manifold bucket.",
    )
    parser.add_argument(
        "--use_neutral_shape",
        action="store_true",
        help="If True, it will assume a NEUTRAL SMPL shape for blending shape params and FK (as in AvatarPoser).",
    )
    parser.add_argument(
        "--out_fps",
        type=int,
        default=60,
        help="Output framerate of the generated data",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="use cpu",
    )
    args = parser.parse_args()

    main(args, device="cpu" if args.cpu else "cuda:0")


if __name__ == "__main__":
    run()
