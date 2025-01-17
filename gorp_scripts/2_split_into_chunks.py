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


def main():
    CHUNK_MAX_FRAMES = 900  # 30 seconds
    DATASET_ROOT = "/data/sandcastle/boxes/fbsource/arvr/projects/kuiper_belt/projects/gorp/GORP_DATASET"
    GORP_AMASS_FOLDER = os.path.join(
        DATASET_ROOT, "GORP_AMASS_FORMAT_v2"
    )  # there must be test_split and train_split
    TARGET_FOLDER = os.path.join(DATASET_ROOT, "GORP_AMASS_FORMAT_v2_chunks")

    print("Processing GORP dataset...")
    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)

    # read train_split.txt and test_split.txt
    train_split_path = os.path.join(GORP_AMASS_FOLDER, "train_split.txt")
    test_split_path = os.path.join(GORP_AMASS_FOLDER, "test_split.txt")
    with open(train_split_path, "r") as f:
        train_split = f.read().splitlines()
    with open(test_split_path, "r") as f:
        test_split = f.read().splitlines()

    # open new files to write the chunks in TARGET_FOLDER
    train_new_split_path = os.path.join(TARGET_FOLDER, "train_split.txt")
    test_new_split_path = os.path.join(TARGET_FOLDER, "test_split.txt")
    train_new_split_file = open(train_new_split_path, "w")
    test_new_split_file = open(test_new_split_path, "w")

    total_frames = {"train": 0, "test": 0}
    for split, filenames, file in (
        ("train", train_split, train_new_split_file),
        ("test", test_split, test_new_split_file),
    ):
        print(f"Processing {split} split...")
        for filename in tqdm(filenames):
            # load the data
            filename = filename[5:]  # remove GORP prefix
            subject, folder_name = filename.split("/")
            folder_name = folder_name.split(".")[0]  # remove .npz extension
            data = np.load(os.path.join(GORP_AMASS_FOLDER, filename), allow_pickle=True)

            keys_to_slice = {
                "trans",
                "poses",
                "root_orient",
                "pose_body",
                "pose_hand",
                "pose_jaw",
                "pose_eye",
            }
            num_frames = data["poses"].shape[0]
            for i in range(0, num_frames, CHUNK_MAX_FRAMES):
                sliced_data = {}
                for key in keys_to_slice:
                    sliced_data[key] = data[key][
                        i : min(i + CHUNK_MAX_FRAMES, num_frames)
                    ]
                # save other keys
                for key in data.keys():
                    if key not in keys_to_slice and key != "tracking_data":
                        sliced_data[key] = data[key]
                    elif key == "tracking_data":
                        # slice all subkeys
                        sliced_data[key] = {}
                        tracking_data = data[key].item()
                        for subkey in tracking_data.keys():
                            sliced_data[key][subkey] = tracking_data[subkey][
                                i : min(i + CHUNK_MAX_FRAMES, num_frames)
                            ]

                subject_path = os.path.join(TARGET_FOLDER, f"{subject}")
                if not os.path.exists(subject_path):
                    os.makedirs(subject_path)
                npz_filename = os.path.join(subject_path, f"{folder_name}_{i:03d}.npz")
                np.savez(
                    npz_filename,
                    **sliced_data,
                )

                # write to new split file
                file.write(f"GORP/{subject}/{folder_name}_{i:03d}.npz\n")

                # update total frames
                total_frames[split] += sliced_data["poses"].shape[0]

    # close
    train_new_split_file.close()
    test_new_split_file.close()
    print(total_frames)
    print("Done!")


if __name__ == "__main__":
    main()
