import os
import pickle
import random
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch as th
from body.data.ossdk_vrs.reader_from_bindings import (
    from_controllers_to_wrists,
    from_headset_to_head,
    InterpInfo,
    OssdkVrsBindingReader,
)
from tqdm import tqdm
from utils.momentum_util import (
    transform_headset_rotation_from_momentum_to_smpl,
    transform_hmd_from_momentum_to_smpl,
    transform_lhand_rotation_from_momentum_to_smpl,
    transform_trajectory_from_momentum_to_smpl,
)
from utils.rotation_conversions import matrix_to_quaternion


def get_folder_name(subject: str, session_name: str) -> str:
    return f"{subject}_{session_name}".lower()


def apply_transforms(npz_dict: dict) -> dict:
    """
    - trans: [seq_len, 3] (z is up)
    """
    # move sequence in the xy plane to the origin
    npz_dict["trans"][:, [0, 1]] -= npz_dict["trans"][0, [0, 1]]
    return npz_dict


def synchronize_with_tracking(
    data: np.ndarray,
    tracking_timestamps: List[float],
    markers_timestamps: List[float],
) -> dict:
    """
    Synchronizes the data with the tracking timestamps.
    """
    assert len(tracking_timestamps) <= len(
        markers_timestamps
    ), "Tracking data is longer than markers data!"
    # get the closest timestamp in the markers data
    interpolated_data = np.zeros((len(tracking_timestamps), *data.shape[1:]))
    for i, time in enumerate(tracking_timestamps):
        interpolation_info = InterpInfo.init_from_sorted_list(
            sorted_list=markers_timestamps, val=time
        )
        factor = interpolation_info.factor
        higher = data[min(interpolation_info.ind_higher, len(data) - 1)]
        lower = data[min(interpolation_info.ind_lower, len(data) - 1)]
        interpolated_data[i] = factor * higher + (1 - factor) * lower
    return interpolated_data


def compute_splits(
    all_names_to_lengths: dict[str, dict[str, dict[str, int]]],
    test_percentage: float = 0.2,
) -> Tuple[List[str], List[str]]:
    """
    computes the train/test splits.
    all_names_to_lengths:
    {
        subject:{
            controllers_pointers: {
                pointer (name): length,
                ...
            }
            tracking_pointers: {
                pointer: length,
                ...
            }
        },
        ...
    }
    """
    all_subjects = sorted(all_names_to_lengths.keys())
    print(f"Total subjects: {all_subjects}")
    # random subjects splits
    random.shuffle(all_subjects)
    num_test_subjects = round(float(len(all_subjects)) * test_percentage)
    test_subjects = all_subjects[:num_test_subjects]
    train_subjects = all_subjects[num_test_subjects:]
    print(f"[TEST] --> {test_subjects}, [TRAIN] --> {train_subjects}")

    # ========================== TRAIN ========================
    train_controllers_pointers = []
    train_tracking_pointers = []
    train_controllers_frames = 0
    train_tracking_frames = 0
    for subject in train_subjects:
        for pointer in all_names_to_lengths[subject]["controllers_pointers"]:
            train_controllers_pointers.append(pointer)
            train_controllers_frames += all_names_to_lengths[subject][
                "controllers_pointers"
            ][pointer]
        for pointer in all_names_to_lengths[subject]["tracking_pointers"]:
            train_tracking_pointers.append(pointer)
            train_tracking_frames += all_names_to_lengths[subject]["tracking_pointers"][
                pointer
            ]
    # ========================== TEST ========================
    test_controllers_pointers = []
    test_tracking_pointers = []
    test_controllers_frames = 0
    test_tracking_frames = 0
    for subject in test_subjects:
        for pointer in all_names_to_lengths[subject]["controllers_pointers"]:
            test_controllers_pointers.append(pointer)
            test_controllers_frames += all_names_to_lengths[subject][
                "controllers_pointers"
            ][pointer]
        for pointer in all_names_to_lengths[subject]["tracking_pointers"]:
            test_tracking_pointers.append(pointer)
            test_tracking_frames += all_names_to_lengths[subject]["tracking_pointers"][
                pointer
            ]

    print(
        f"[CONTROLLERS] Duration distribution in test: {round(test_controllers_frames / (test_controllers_frames + train_controllers_frames) * 100, 1)}%"
    )
    print(
        f"[TRACKING] Duration distribution in test: {round(test_tracking_frames / (test_tracking_frames + train_tracking_frames) * 100, 1)}%"
    )
    return (
        train_controllers_pointers + train_tracking_pointers,
        test_controllers_pointers + test_tracking_pointers,
    )


def store_split(filenames: List[str], output_file: str) -> None:
    # open a file for writing
    with open(output_file, "w") as f:
        # write each string in the list to the file
        for name in filenames:
            f.write(name + ".npz\n")


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
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Get the wrist positions from the hands tracking data
    Returns:
        tracking_inputs (th.Tensor): [n_frames, 3, 7] --> [trans_x, transy, trans_z, rot_w, rot_x, rot_y, rot_z]
        confidences (th.Tensor): [n_frames, 2] --> [left_hand_confidence, right_hand_confidence]
        tracked (th.Tensor): [n_frames, 2] --> [left_hand_confidence, right_hand_confidence
    """
    all_head_wrists = th.zeros((vrs_reader.n_frames, 3, 7))
    all_confidences = th.zeros((vrs_reader.n_frames, 2))
    all_tracked = th.zeros((vrs_reader.n_frames, 2)).bool()
    zero_hands = [0] * 7
    for frame_idx in tqdm(range(vrs_reader.n_frames)):
        headset = vrs_reader.get_headset_from_frame_index(frame_idx)
        head = from_headset_to_head(
            headset
        )  # NOTE: this is very slow!! 40 milliseconds per frame. Should be parallelized for all frames.
        if vrs_reader.tracking_inputs[frame_idx].hands is not None:
            (
                wrist_left,
                wrist_right,
            ) = vrs_reader.get_hands_tracking_from_frame_index(frame_idx)
            all_head_wrists[frame_idx] = th.Tensor((head, wrist_left, wrist_right))
            confidences = vrs_reader.get_hands_confidence_from_frame_index(frame_idx)
            all_confidences[frame_idx] = th.Tensor(confidences)
            tracked = vrs_reader.are_hands_tracked_from_frame_index(frame_idx)
            all_tracked[frame_idx] = th.Tensor(tracked)
        else:
            all_head_wrists[frame_idx] = th.Tensor((head, zero_hands, zero_hands))
            all_confidences[frame_idx] = th.Tensor((0, 0))
            all_tracked[frame_idx] = th.Tensor((0, 0))

    return all_head_wrists, all_confidences, all_tracked


def get_head_wrist_from_controllers(
    vrs_reader: OssdkVrsBindingReader,
) -> th.Tensor:
    head_wrists = []
    for idx in range(vrs_reader.n_frames):
        headset = vrs_reader.get_headset_from_frame_index(idx)
        inputs = vrs_reader.tracking_inputs
        assert inputs[idx].controllers.left.in_hand, "Left controller is not in hand!"  # pyre-ignore
        assert inputs[idx].controllers.right.in_hand, "Right controller is not in hand!"  # pyre-ignore
        (
            controller_left,
            controller_right,
        ) = vrs_reader.get_controllers_from_frame_index(idx)
        head = from_headset_to_head(headset)
        wrist_left, wrist_right = from_controllers_to_wrists(
            controller_left=controller_left,
            controller_right=controller_right,
        )
        head_wrists.append((head, wrist_left, wrist_right))
    return th.Tensor(head_wrists)


def get_tracking_data(
    vrs_reader: OssdkVrsBindingReader,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, bool]:
    """
    Returns the head and wrist positions from the tracking data
    Returns:
        head_wrists (th.Tensor): [n_frames, 3, 7] --> [trans_x, trans_y, trans_z, rot_w, rot_x, rot_y, rot_z]
        confidences (th.Tensor): [n_frames, 2] --> [left_hand_confidence, right_hand_confidence]
        available (th.Tensor): [n_frames, 2] --> [left_hand_confidence, right_hand_confidence]
        is_hands_tracking (bool): True if the hands tracking data is available, False otherwise
    """
    if not is_hands_tracking_sample(vrs_reader):
        # retrieve controllers data
        tracking_signal = get_head_wrist_from_controllers(vrs_reader)  # [frames, 3, 7]
        confidences = th.ones((tracking_signal.shape[0], 2))  # [frames, 2]
        available = th.ones((tracking_signal.shape[0], 2))  # [frames, 2]
        return tracking_signal, confidences, available, False
    tracking_signal, confidences, available = get_head_wrist_from_hands_tracking(
        vrs_reader
    )
    return tracking_signal, confidences, available, True


def main():
    DATASET_ROOT = "/data/sandcastle/boxes/fbsource/arvr/projects/kuiper_belt/projects/gorp/GORP_DATASET"
    SMPL_DATA_PATH = os.path.join(DATASET_ROOT, "GORP_SMPL_OutputMoshpp")
    MARKERS_DATA_PATH = os.path.join(
        DATASET_ROOT,
        "GORP_ORIGINAL_NPZ_C3D_INPUTSNPZ_OutputGorpScripts_v4controllers",
    )
    csv_path = os.path.join(DATASET_ROOT, "dataset.csv")
    TARGET_FOLDER = os.path.join(DATASET_ROOT, "GORP_AMASS_FORMAT_v2")
    print("Processing GORP dataset...")
    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)

    df = pd.read_csv(csv_path)
    df["FolderName"] = df.apply(
        lambda row: get_folder_name(row["Subject"], row["Motion"]), axis=1
    )
    df = df[~df["FolderName"].str.contains("-test")]  # we discarded them.
    df = df[~df["FolderName"].str.contains("_rom")]  # we discarded them.

    missing_folders = []
    lengths = {
        "pilot": [],
        "main": [],
        "controllers": [],
        "tracking": [],
    }
    all_names_to_lengths = {}
    for i, row in tqdm(df.iterrows()):
        folder_name = row["FolderName"]
        smpl_folder = os.path.join(SMPL_DATA_PATH, folder_name)
        smpl_filename = os.path.join(smpl_folder, "marker_data_stageii.pkl")
        tracking_folder = os.path.join(MARKERS_DATA_PATH, folder_name)
        tracking_filename = os.path.join(tracking_folder, "vr_data.npz")
        if not os.path.exists(tracking_filename):
            print(
                f"[IMPORTANT] {folder_name} - tracking data does not exist. Skipping..."
            )
            missing_folders.append((folder_name, "tracking"))
            continue
        if not os.path.exists(smpl_folder):
            print(f"{folder_name} - Folder does not exist")
            missing_folders.append((folder_name, "smpl"))
            continue
        elif not os.path.exists(smpl_filename):
            print(f"[IMPORTANT] {folder_name} - marker_data_stageii.pkl does not exist")
            missing_folders.append((folder_name, "smpl"))
            continue

        # load SMPL data
        with open(smpl_filename, "rb") as f:
            data = pickle.load(f)

        # load TRACKING data
        try:
            tracking_data = dict(np.load(tracking_filename))
        except Exception as e:
            print(f"[IMPORTANT] {folder_name} - tracking data is corrupted!")
            missing_folders.append((folder_name, "tracking corrupted"))
            continue
        tracking_timestamps = tracking_data[
            "deviceTimestamps"
        ]  # same as handsTimestamps
        if "handsTimestamps" in tracking_data and len(
            tracking_data["handsTimestamps"]
        ) < len(tracking_timestamps):
            # The hand tracking data often starts a few frames later than the tracking data --> Ignore first deviceTimestamps
            offset = 1
            while tracking_timestamps[offset] != tracking_data["handsTimestamps"][0]:
                offset += 1
                print(f"offset: {offset}")
                if offset > len(tracking_timestamps):
                    raise ValueError(
                        "Tracking data is not synchronized with hands data!"
                    )
            assert tracking_timestamps[offset] == tracking_data["handsTimestamps"][0]
            tracking_data["deviceTimestamps"] = tracking_data["deviceTimestamps"][
                offset:
            ]
            for key in tracking_data:
                if "hmd" in key or "lCon" in key or "rCon" in key:
                    tracking_data[key] = tracking_data[key][offset:]

        # load markers data
        markers_data_file = os.path.join(
            MARKERS_DATA_PATH, folder_name, "marker_data.npz"
        )
        smpl_timestamps = np.load(markers_data_file)["markerTimestamps"]

        # markers from OptiTrack (SMPL) sometimes start a few frames later than the tracking data. We only keep the frames where we have GT markers.
        init_padding = 0
        end_padding = 0
        while tracking_timestamps[init_padding] < smpl_timestamps[0]:
            # tracking_timestamps = tracking_timestamps[1:]
            init_padding += 1
        # update the tracking data slicing out the init padding
        for key in tracking_data:
            tracking_data[key] = tracking_data[key][init_padding:]
        while tracking_timestamps[-end_padding - 1] > smpl_timestamps[-1]:
            # tracking_timestamps = tracking_timestamps[:-1]
            end_padding += 1
        # update the tracking data slicing out the end padding
        if end_padding > 0:
            for key in tracking_data:
                tracking_data[key] = tracking_data[key][:-end_padding]

        tracking_timestamps = tracking_data[
            "deviceTimestamps"
        ]  # update with the new tracking timestamps
        assert (
            smpl_timestamps[0] <= tracking_timestamps[0]
        ), f"Tracking starts before markers data! {smpl_timestamps[0]} vs {tracking_timestamps[0]}"
        assert (
            smpl_timestamps[-1] >= tracking_timestamps[-1]
        ), f"Tracking ends after markers data! {smpl_timestamps[-1]} vs {tracking_timestamps[-1]}"

        # =================== SYNCHRONIZE TRACKING AND SMPL DATA ===================
        KEYS_TO_SYNC = [
            "leftHandConfidence",
            "rightHandConfidence",
            "leftHandTransform",
            "rightHandTransform",
            "hmdTransforms",
            "lConValid",
            "lConTransforms",
            "rConValid",
            "rConTransforms",
        ]
        # get the controllers and tracking data
        nframes = len(tracking_timestamps)
        mismatch = False
        for key in KEYS_TO_SYNC:
            if key not in tracking_data:
                continue

            if "Transform" in key:
                # from 4x4 to [trans_x, trans_y, trans_z, rot_w, rot_x, rot_y, rot_z]
                trans = torch.Tensor(tracking_data[key][:, 3, :3])  # [n_frames, 3]
                # trans = transform_trajectory_from_momentum_to_smpl(trans)

                quat = matrix_to_quaternion(
                    torch.Tensor(tracking_data[key][:, :3, :3])
                )  # [n_frames, 4]
                """
                if key == "hmdTransforms":
                    # transform the inputs from momentum to smpl
                    quat = transform_headset_rotation_from_momentum_to_smpl(quat)
                elif key in ["leftHandTransform", "rightHandTransform"]:
                    # transform the inputs from momentum to smpl
                    quat = transform_lhand_rotation_from_momentum_to_smpl(quat)
                """
                tracking_data[key] = np.concatenate(
                    (trans.numpy(), quat.numpy()), axis=-1
                )
            # assert (
            #    tracking_data[key].shape[0] == nframes
            # ), f"[{folder_name}] {key} has {tracking_data[key].shape[0]} frames instead of {nframes}"
            if tracking_data[key].shape[0] != nframes:
                mismatch = True
        if mismatch:
            print(f"[IMPORTANT] {folder_name} - mismatch in tracking data")
            continue
        tracking_data_to_store = {
            key: tracking_data[key] for key in KEYS_TO_SYNC if key in tracking_data
        }
        # Synchronize the data with the tracking timestamps.
        fullpose = synchronize_with_tracking(
            data["fullpose"], tracking_timestamps, smpl_timestamps
        )  # [seq_len, 165]
        trans = synchronize_with_tracking(
            data["trans"], tracking_timestamps, smpl_timestamps
        )  # [seq_len, 3]

        # =================== STORING DATA IN AMASS FORMAT FOR STANDARDIZATION ===========================
        npz_dict = {
            "gender": "neutral",  # "neutral"
            "surface_model_type": "smplx",  # "smplx"
            "mocap_frame_rate": round(
                data["stageii_debug_details"]["mocap_frame_rate"], 1
            ),  # 120
            "mocap_time_length": data["stageii_debug_details"][
                "mocap_time_length"
            ],  # float
            "markers_latent": data["markers_latent"],  # [NUM_MARKERS, 3]
            "latent_labels": data["latent_labels"],  # labels of the markers_latent
            "markers_latent_vids": data[
                "markers_latent_vids"
            ],  # locators of the markers
            "trans": trans,  # [seq_len, 3]
            "poses": fullpose,  # [seq_len, 165]
            "betas": data["betas"][:16],  # [16]
            "num_betas": 16,  # 16
            "root_orient": fullpose[:, :3],  # [seq_len, 3]
            "pose_body": fullpose[:, 3:66],  # [seq_len, 63]
            "pose_hand": fullpose[:, 75:],  # [seq_len, 90]
            "pose_jaw": fullpose[:, 66:69],  # [seq_len, 3]
            "pose_eye": fullpose[:, 69:75],  # [seq_len, 6]
            # === tracking data ===
            "tracking_data": tracking_data_to_store,  # dict
        }

        # apply transforms
        # npz_dict = apply_transforms(npz_dict)

        subject = row["Subject"]
        subject_path = os.path.join(TARGET_FOLDER, f"{subject}")
        if not os.path.exists(subject_path):
            os.makedirs(subject_path)
        npz_filename = os.path.join(subject_path, f"{folder_name}.npz")
        np.savez(
            npz_filename,
            **npz_dict,
        )
        # else:
        #    print("Already exists:", npz_filename)

        pointer = f"GORP/{subject}/{folder_name}"
        length_minutes = round(data["fullpose"].shape[0] / 120 / 60, 1)
        if "_rom" not in folder_name:
            if "ps" in folder_name:
                lengths["pilot"].append(length_minutes)
            else:
                lengths["main"].append(length_minutes)

            if subject not in all_names_to_lengths:
                all_names_to_lengths[subject] = {
                    "tracking_pointers": {},
                    "controllers_pointers": {},
                }
            if "handtracking" in folder_name:
                lengths["tracking"].append(length_minutes)
                all_names_to_lengths[subject]["tracking_pointers"][
                    pointer
                ] = length_minutes
            elif "controllers" in folder_name:
                lengths["controllers"].append(length_minutes)
                all_names_to_lengths[subject]["controllers_pointers"][
                    pointer
                ] = length_minutes
            else:
                raise ValueError("Unknown type of sequence")

    print(f"Missing folders: {missing_folders}")

    """
    for i in range(50):
        print(i, "=" * 50)
        # store splits (can run multiple times until you get a good proportion). Goal is to have 20/80 percent for both controllers/hands tracking.
        train_names, test_names = compute_splits(
            all_names_to_lengths, test_percentage=0.2
        )
        store_split(train_names, os.path.join(TARGET_FOLDER, f"train_split_{i}.txt"))
        store_split(test_names, os.path.join(TARGET_FOLDER, f"test_split_{i}.txt"))
    """

    def print_summary(title: str, lengths: List[int]):
        print(title)
        print("Total number of sequences:", len(lengths))
        print("Total number of minutes:", sum(lengths))
        print("Average number of minutes per sequence:", np.mean(lengths))

    # print summary of the dataset
    print("=======================")
    print("Summary of the dataset split into Controllers/Tracking (minutes):")
    print("=======================")
    print_summary("Controllers dataset:", lengths["controllers"])
    print("=======================")
    print_summary("Tracking dataset:", lengths["tracking"])
    print("=======================")
    print("\n\nSummary of the dataset split into Pilot/Main (minutes):")
    print("=======================")
    print_summary("Main dataset:", lengths["pilot"])
    print("=======================")
    print_summary("Pilot dataset:", lengths["main"])
    print("=======================")
    print_summary("COMPLETE dataset:", lengths["main"] + lengths["pilot"])
    print("=======================")


if __name__ == "__main__":
    main()
