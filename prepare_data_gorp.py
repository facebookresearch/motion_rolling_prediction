# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import os

import numpy as np
import torch
from evaluation.utils import BodyModelsWrapper
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from loguru import logger
from tqdm import tqdm
from utils import utils_transform
from utils.constants import SMPLGenderParam, SMPLModelType
from utils.gorp_util import (
    transform_controllers_from_gorp_to_smpl,
    transform_hand_tracking_from_gorp_to_smpl,
)
from utils.rotation_conversions import quaternion_to_matrix
from pathlib import Path

body_model_type = SMPLModelType.SMPLX

def from_smpl_to_input_features(
    smpl_pose_vec: torch.Tensor, pose_joints_world: torch.Tensor, kintree
) -> dict:
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


def from_controllers_to_hmd_conditioning(hmd_tracking):
    """
    Receives a tensor of shape [frames, 3, 7] containing the hmd tracking in SMPL format
    for [head, left_controller, right_controller]. Returns the features used
    as conditioning for our models.

    The seven elements of the hmd_tracking are in Momentum format:
    - position: [x, y, z]
    - rotation: [rw, rx, ry, rz]
    """
    # changing the scale from cm to m
    global_hmd_pos = hmd_tracking[..., :3]
    global_hmd_quat = hmd_tracking[..., 3:7]

    global_hmd_rot = quaternion_to_matrix(global_hmd_quat)
    num_frames = global_hmd_rot.shape[0] - 1
    rotation_velocity_global_matrot = torch.matmul(
        torch.inverse(global_hmd_rot[:-1]), global_hmd_rot[1:]
    )
    rotation_velocity_global_6d = utils_transform.matrot2sixd(
        rotation_velocity_global_matrot.reshape(-1, 3, 3)
    ).reshape(rotation_velocity_global_matrot.shape[0], -1, 6)
    input_rotation_velocity_global_6d = rotation_velocity_global_6d.reshape(
        num_frames, -1
    )

    rotation_global_6d = utils_transform.matrot2sixd(
        global_hmd_rot.reshape(-1, 3, 3)
    ).reshape(global_hmd_rot.shape[0], -1, 6)
    input_rotation_global_6d = rotation_global_6d[1:, :, :].reshape(num_frames, -1)

    position_global_full_gt_world = (
        global_hmd_pos  # position of joints relative to the world origin
    )
    input_translation_global = position_global_full_gt_world[1:].reshape(num_frames, -1)
    input_translation_velocity_global = (
        position_global_full_gt_world[1:] - position_global_full_gt_world[:-1]
    ).reshape(num_frames, -1)

    hmd_position_global_full_gt_list = torch.cat(
        [
            input_rotation_global_6d,
            input_rotation_velocity_global_6d,
            input_translation_global,
            input_translation_velocity_global,
        ],
        dim=-1,
    )
    return (
        hmd_position_global_full_gt_list,
        global_hmd_pos[1:],
        global_hmd_rot[1:],
    )


def get_downsample_idces(num_frames, fps, target_fps):
    if target_fps > fps:
        raise AssertionError("Cannot supersample data!", target_fps, fps)
    else:
        fps_ratio = float(target_fps) / fps
        new_num_frames = int(fps_ratio * num_frames)
        downsamp_inds = np.linspace(0, num_frames - 1, num=new_num_frames, dtype=int)
    return downsamp_inds


def main(args, device="cuda:0"):
    # list folders of args.splits_dir
    body_model = None
    all_datasets = sorted(os.listdir(args.splits_dir))
    for dataroot_subset in all_datasets:
        for phase in ["test_controllers", "test_tracking", "train"]:
            # all_offsets = []
            logger.info(f"Processing {dataroot_subset} {phase}...")
            split_file = args.splits_dir / dataroot_subset / (phase + "_split.txt")
            if not split_file.exists():
                logger.info(f"{split_file} does not exist, skipping...")
                continue

            savedir = args.save_dir / dataroot_subset / phase
            savedir.mkdir(parents=True, exist_ok=True)

            with open(split_file, "r") as f:
                filepaths = [line.strip() for line in f]

            start_idx = 0
            if "tracking" in phase:
                start_idx = 147
            for idx, filepath in tqdm(enumerate(filepaths)):
                dst_fname = "{}.pt".format(start_idx + idx + 1)
                current_file_path = args.root_dir / filepath
                assert current_file_path.exists(), f"{current_file_path} does not exist. Aborting..."
                bdata = np.load(
                    current_file_path,
                    allow_pickle=True,
                )
                fps = 30  # ALWAYS FOR GORP
                tr_data = bdata["tracking_data"].item()  # dict
                headset_transform = tr_data["hmdTransforms"]
                headset_transform = torch.Tensor(headset_transform)
                smpl_inputs_3p = [
                    headset_transform,
                ]
                if "handtracking" not in filepath.lower():
                    # ================ CONTROLLERS ================
                    controller_left = tr_data["lConTransforms"]
                    controller_right = tr_data["rConTransforms"]
                    smpl_inputs_3p.append(torch.Tensor(controller_left))
                    smpl_inputs_3p.append(torch.Tensor(controller_right))
                    lhand_conf = (
                        np.ones(headset_transform.shape[0]) * tr_data["lConValid"]
                    )
                    rhand_conf = (
                        np.ones(headset_transform.shape[0]) * tr_data["rConValid"]
                    )
                    smpl_inputs_3p = torch.stack(
                        smpl_inputs_3p, dim=1
                    )  # [frames, 3, 7]
                    smpl_inputs_3p = transform_controllers_from_gorp_to_smpl(
                        smpl_inputs_3p
                    )
                else:
                    # ================ HANDS TRACKING ================
                    smpl_inputs_3p.append(torch.Tensor(tr_data["leftHandTransform"]))
                    smpl_inputs_3p.append(torch.Tensor(tr_data["rightHandTransform"]))
                    lhand_conf = tr_data["leftHandConfidence"]
                    rhand_conf = tr_data["rightHandConfidence"]
                    smpl_inputs_3p = torch.stack(
                        smpl_inputs_3p, dim=1
                    )  # [frames, 3, 7]
                    smpl_inputs_3p = transform_hand_tracking_from_gorp_to_smpl(
                        smpl_inputs_3p
                    )

                downsample_idces = get_downsample_idces(
                    smpl_inputs_3p.shape[0], fps, args.out_fps
                )
                smpl_inputs_3p = smpl_inputs_3p[downsample_idces]
                real_hmd_cond, real_global_hmd_pos, real_global_hmd_rot = (
                    from_controllers_to_hmd_conditioning(smpl_inputs_3p)
                )
                # TODO: here velocities will be wrong for frames with neighboring frames with no hands tracking!!

                # ============= ADD GT ============
                bdata_poses = bdata["poses"][downsample_idces, ...]
                bdata_trans = bdata["trans"][downsample_idces, ...]

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
                    "betas": torch.Tensor(bdata["betas"][:16]).repeat(
                        bdata_poses.shape[0], 1
                    ),
                }
                if body_model is None:
                    logger.info("Initializing body model: {}".format(body_model_type))
                    body_model = BodyModelsWrapper(args.support_dir)
                body_pose_world = body_model(
                    {k: v.to(device) for k, v in body_parms.items()},
                    body_model_type,
                    SMPLGenderParam.NEUTRAL,
                )
                gt_joints_world_space = body_pose_world.Jtr[
                    :, :22, :
                ].cpu()  # position of joints relative to the world origin

                kintree = body_model.get_kin_tree(
                    body_model_type, SMPLGenderParam.NEUTRAL
                )
                data = from_smpl_to_input_features(
                    bdata_poses,
                    gt_joints_world_space,
                    kintree,
                )

                num_frames = real_hmd_cond.shape[0]
                if num_frames == 0:
                    logger.info("No frames found in {}".format(filepath))
                    continue

                data["left_hand_confidence"] = lhand_conf[downsample_idces][1:]
                data["right_hand_confidence"] = rhand_conf[downsample_idces][1:]
                data["hmd_position_global_full_real_list"] = real_hmd_cond
                data["framerate"] = args.out_fps
                data["gender"] = "neutral"
                data["surface_model_type"] = SMPLModelType.SMPLX
                data["filepath"] = filepath
                data["body_parms_list"] = body_parms

                torch.save(data, savedir / dst_fname)


def run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("./prepare_data/gorp"),
        help="=dir where the data splits are defined",
    )
    parser.add_argument(
        "--support_dir",
        type=Path,
        default=Path("./SMPL"),
        help="=dir where you put your smplh and dmpls dirs",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("./datasets_processed/gorp/new_format_data"),
        help="=dir where you want to save your generated data",
    )
    parser.add_argument(
        "--root_dir", type=Path, required=True, help="=dir where you put your AMASS data"
    )
    parser.add_argument(
        "--out_fps",
        type=int,
        default=30,
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
