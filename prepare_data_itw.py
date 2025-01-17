# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import os
import tempfile

import numpy as np
import torch
import utils.constants as constants
from body.data.ossdk_vrs.reader_from_bindings import (
    from_controllers_to_wrists,
    from_headset_to_head,
    OssdkVrsBindingReader,
)
from loguru import logger
from tqdm import tqdm
from utils import utils_transform
from utils.config import pathmgr
from utils.constants import DatasetType, SMPLModelType

from utils.momentum_util import (
    transform_hand_tracking_from_itw_to_smpl,
    transform_hmd_from_momentum_to_smpl,
)
from utils.rotation_conversions import quaternion_to_matrix


def is_hands_tracking_sample(
    vrs_reader: OssdkVrsBindingReader,
) -> bool:
    """
    Check if the vrs file contains hand tracking data
    """
    for idx in range(vrs_reader.n_frames):
        if vrs_reader.tracking_inputs[idx].hands is not None:
            return True
    return False


def get_head_wrist_from_hands_tracking(
    vrs_reader: OssdkVrsBindingReader,
    min_confidence: float = 0.8,
) -> torch.Tensor:
    """
    Get the wrist positions from the hands tracking data
    """
    head_wrists = []
    for frame_idx in range(vrs_reader.n_frames):
        headset = vrs_reader.get_headset_from_frame_index(frame_idx)
        (
            wrist_left,
            wrist_right,
        ) = vrs_reader.get_hands_tracking_from_frame_index(frame_idx)
        left_tracked, right_tracked = vrs_reader.are_hands_tracked_from_frame_index(
            frame_idx
        )
        left_conf, right_conf = vrs_reader.get_hands_confidence_from_frame_index(
            frame_idx
        )
        if wrist_left is None or not left_tracked or left_conf < min_confidence:
            wrist_left = torch.zeros(7)
        if wrist_right is None or not right_tracked or right_conf < min_confidence:
            wrist_right = torch.zeros(7)
        head = from_headset_to_head(headset)
        head_wrists.append((head, wrist_left, wrist_right))
    return torch.Tensor(head_wrists)


def get_head_wrist_skel_states_tensor_from_sensors(
    vrs_reader: OssdkVrsBindingReader,
) -> torch.Tensor:
    head_wrists = []
    for frame_idx in range(vrs_reader.n_frames):
        headset = vrs_reader.get_headset_from_frame_index(frame_idx)
        (
            controller_left,
            controller_right,
        ) = vrs_reader.get_controllers_from_frame_index(frame_idx)
        head = from_headset_to_head(headset)
        wrist_left, wrist_right = from_controllers_to_wrists(
            controller_left=controller_left,
            controller_right=controller_right,
        )
        head_wrists.append((head, wrist_left, wrist_right))
    return torch.Tensor(head_wrists)


def get_tracking_data(
    vrs_reader: OssdkVrsBindingReader,
) -> torch.Tensor:
    if not is_hands_tracking_sample(vrs_reader):
        # retrieve controllers data
        return get_head_wrist_skel_states_tensor_from_sensors(vrs_reader)
    return get_head_wrist_from_hands_tracking(vrs_reader)


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
    # replace nan values with 0
    hmd_position_global_full_gt_list[torch.isnan(hmd_position_global_full_gt_list)] = 0
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


def main(args):
    if args.use_vip:
        from iopath import PathManager
        from iopath.fb.manifold import ManifoldPathHandler

        vip_pathmgr = PathManager()
        vip_pathmgr.register_handler(ManifoldPathHandler(use_vip=True))
    else:
        vip_pathmgr = pathmgr

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

            for idx, filepath in tqdm(enumerate(filepaths)):
                dst_fname = "{}.pt".format(idx + 1)
                manifold_path = os.path.join(savedir, dst_fname)
                # if pathmgr.exists(manifold_path):
                #    logger.info(f"File {manifold_path} already exists, skipping...")
                #    continue

                vrs_path = os.path.join(
                    args.root_dir, filepath, "sensor_data_ossdk.vrs"
                )
                logger.info(f"Processing as vrs file: {vrs_path}")
                vrs_reader = OssdkVrsBindingReader(vrs_path)
                fps = round(vrs_reader.fps, 2)  # in case it's 29.99997
                momentum_inputs_3p = get_tracking_data(
                    vrs_reader
                )  # [frames, 3, 7] --> [trans_x, trans_y, trans_z, rot_w, rot_x, rot_y, rot_z]
                downsample_idces = get_downsample_idces(
                    momentum_inputs_3p.shape[0], fps, args.out_fps
                )
                momentum_inputs_3p = momentum_inputs_3p[downsample_idces]
                left_tracked = ~(momentum_inputs_3p[:, 1] == 0).all(axis=1)
                right_tracked = ~(momentum_inputs_3p[:, 2] == 0).all(axis=1)
                """
                if "hand" in filepath.lower():
                    # HT
                    smpl_inputs_3p = transform_controllers_from_gorp_to_smpl(
                        momentum_inputs_3p, scale=1.0 / 100
                    )
                else:
                    # MC
                    smpl_inputs_3p = transform_hand_tracking_from_gorp_to_smpl(
                        momentum_inputs_3p, scale=1.0 / 100
                    )
                """
                smpl_inputs_3p = transform_hmd_from_momentum_to_smpl(
                    momentum_inputs_3p.clone()
                )
                if "hand" in filepath.lower():
                    # HT
                    smpl_inputs_3p_hands = transform_hand_tracking_from_itw_to_smpl(
                        momentum_inputs_3p
                    )
                    smpl_inputs_3p = torch.stack(
                        [
                            smpl_inputs_3p[:, 0],
                            smpl_inputs_3p_hands[:, 1],
                            smpl_inputs_3p_hands[:, 2],
                        ],
                        dim=1,
                    )
                hmd_cond, global_hmd_pos, global_hmd_rot = (
                    from_controllers_to_hmd_conditioning(smpl_inputs_3p)
                )

                # FILTER OUT FRAMES WITH LOW HAND CONFIDENCE OR NOT TRACKED HANDS
                lhand_idces = constants.ENTITIES_IDCES[DatasetType.ITW][1]  # [seq_len]
                rhand_idces = constants.ENTITIES_IDCES[DatasetType.ITW][2]  # [seq_len]
                hmd_cond[:, lhand_idces] = (
                    hmd_cond[:, lhand_idces] * left_tracked[1:, None]
                )
                hmd_cond[:, rhand_idces] = (
                    hmd_cond[:, rhand_idces] * right_tracked[1:, None]
                )

                data = {}

                num_frames = hmd_cond.shape[0]
                if num_frames == 0:
                    logger.info("No frames found in {}".format(filepath))
                    continue

                head_global_trans = torch.eye(4).repeat(num_frames, 1, 1)
                head_global_trans[:, :3, :3] = global_hmd_rot[
                    :, 0, :
                ]  # slice out the head rotation
                head_global_trans[:, :3, 3] = global_hmd_pos[
                    :, 0, :
                ]  # slice out the head position

                data["rotation_local_full_gt_list"] = None
                data["rotation_global_full_gt_list"] = None
                data["hmd_position_global_full_gt_list"] = hmd_cond
                data["head_global_trans_list"] = head_global_trans
                data["position_global_full_gt_world"] = None
                data["body_parms_list"] = {
                    "betas": torch.zeros(num_frames + 1, 16),
                }
                data["framerate"] = args.out_fps
                data["gender"] = "neutral"
                data["surface_model_type"] = SMPLModelType.SMPLX
                data["filepath"] = filepath

                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmp_local_path = os.path.join(tmpdirname, dst_fname)
                    torch.save(data, tmp_local_path)
                    pathmgr.copy(
                        src_path=tmp_local_path, dst_path=manifold_path, overwrite=True
                    )


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
        "--out_fps",
        type=int,
        default=60,
        help="Output framerate of the generated data",
    )
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    run()
