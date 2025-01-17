# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import os

import numpy as np
from loguru import logger
from tqdm import tqdm
from utils.config import pathmgr


def main(args):
    THRESHOLD = 0.8
    # list folders of args.splits_dir
    all_datasets = sorted(os.listdir(args.splits_dir))
    info = {
        "metadata": {
            "dataset_name": "gorp",
            "eval_name": "real_input_loss",
            "masker": "seg_hands_idp",
        }
    }
    for dataroot_subset in all_datasets:
        all_gaps = {}
        for phase in [
            "test",
        ]:
            total_frames = 0
            # all_offsets = []
            logger.info(f"Processing {dataroot_subset} {phase}...")
            split_file = os.path.join(
                args.splits_dir, dataroot_subset, phase + "_split.txt"
            )
            if not pathmgr.exists(split_file):
                logger.info(f"{split_file} does not exist, skipping...")
                continue

            with open(split_file, "r") as f:
                filepaths = [line.strip() for line in f]

            for idx, filepath in tqdm(enumerate(filepaths)):
                bdata = np.load(
                    pathmgr.get_local_path(os.path.join(args.root_dir, filepath)),
                    allow_pickle=True,
                )
                tr_data = bdata["tracking_data"].item()  # dicts
                if "handtracking" not in filepath.lower():
                    lhand_conf = tr_data["leftHandConfidence"]
                    rhand_conf = tr_data["rightHandConfidence"]
                    # print(filepath, lhand_conf.mean(), rhand_conf.mean())
                    all_gaps[f"GORP-{idx+1}"] = [[], []]
                    continue  # not interested in controllers
                else:
                    lhand_conf = tr_data["leftHandConfidence"]
                    rhand_conf = tr_data["rightHandConfidence"]

                    total_frames -= 1
                    lhand_conf = lhand_conf[
                        1:
                    ]  # remove first frame as in preprocessing of data
                    rhand_conf = rhand_conf[
                        1:
                    ]  # remove first frame as in preprocessing of data

                    def get_gaps_per_hand(conf, threshold):
                        zero_segments = []
                        conf = conf.copy()
                        conf[conf < threshold] = 0
                        t0 = -1
                        for i in range(conf.shape[0]):
                            if conf[i] == 0 and t0 == -1:
                                t0 = i  # start of a zero segment
                            elif conf[i] > 0 and t0 != -1:
                                zero_segments.append((t0, i))
                                t0 = -1
                        return zero_segments

                    lhand_file_gaps = get_gaps_per_hand(lhand_conf, THRESHOLD)
                    rhand_file_gaps = get_gaps_per_hand(rhand_conf, THRESHOLD)
                    print("=======")
                    print(filepath)
                    print(f"GORP-{idx+1}", lhand_file_gaps, rhand_file_gaps)

                    all_gaps[f"GORP-{idx+1}"] = [lhand_file_gaps, rhand_file_gaps]
    info["gaps"] = all_gaps

    import json

    with pathmgr.open(
        "manifold://xr_body/tree/personal/gbarquero/datasets/agrol/GORP/eval_gap_configs/real_input.json",
        "w",
    ) as f:
        # save json
        json.dump(
            info,
            f,
            indent=4,
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
        "--root_dir", type=str, default=None, help="=dir where you put your AMASS data"
    )
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    run()
