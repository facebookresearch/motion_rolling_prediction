# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import os

import numpy as np
from loguru import logger
from tqdm import tqdm
from utils.config import pathmgr


def main(args):
    # list folders of args.splits_dir
    all_datasets = sorted(os.listdir(args.splits_dir))
    for dataroot_subset in all_datasets:
        for phase in ["test", "train"]:
            frames_count = 0
            all_gaps_lhand_by_threshold = {
                0.5: [],
                0.80: [],
                0.9: [],
                0.95: [],
                0.975: [],
                0.99: [],
            }
            all_gaps_rhand_by_threshold = {
                k: [] for k in all_gaps_lhand_by_threshold.keys()
            }

            total_frames = 0
            all_subjects = set()
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
                # if pathmgr.exists(manifold_path):
                #    logger.info(f"File {manifold_path} already exists, skipping...")
                #    continue
                bdata = np.load(
                    pathmgr.get_local_path(os.path.join(args.root_dir, filepath)),
                    allow_pickle=True,
                )
                total_frames += (
                    bdata["tracking_data"].item()["leftHandConfidence"].shape[0]
                )
                subject = filepath.split("/")[1]
                all_subjects.add(subject)
                tr_data = bdata["tracking_data"].item()  # dict
                if "handtracking" not in filepath.lower():
                    lhand_conf = tr_data["leftHandConfidence"]
                    rhand_conf = tr_data["rightHandConfidence"]
                    print(filepath, lhand_conf.mean(), rhand_conf.mean())
                    continue  # not interested in controllers
                else:
                    lhand_conf = tr_data["leftHandConfidence"]
                    rhand_conf = tr_data["rightHandConfidence"]

                    total_frames += lhand_conf.shape[0]

                    def get_gap_lengths_per_hand(conf, threshold):
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
                        return [seg[1] - seg[0] for seg in zero_segments]

                    for threshold in all_gaps_lhand_by_threshold.keys():
                        all_gaps_lhand_by_threshold[threshold].extend(
                            get_gap_lengths_per_hand(lhand_conf, threshold)
                        )
                        all_gaps_rhand_by_threshold[threshold].extend(
                            get_gap_lengths_per_hand(rhand_conf, threshold)
                        )
            from matplotlib import pyplot as plt

            # plot the distribution of gaps for each threshold in a subplots
            fig, axs = plt.subplots(
                1, len(all_gaps_lhand_by_threshold), figsize=(20, 5)
            )
            for i, (threshold, gaps) in enumerate(all_gaps_lhand_by_threshold.items()):
                axs[i].hist(gaps, bins=100)
                axs[i].set_title(
                    f"Threshold: {threshold}, Mean: {round(np.mean(gaps), 2)}"
                )
            plt.savefig(f"gap_distribution_{phase}.png")
            # print percentage of frames with no hand tracking
            for threshold in all_gaps_lhand_by_threshold.keys():
                print(
                    f"Percentage of frames with no left-hand tracking for threshold {threshold}: {sum([g for g in all_gaps_lhand_by_threshold[threshold] if g > 0]) / total_frames * 100:.2f}%"
                )
            for threshold in all_gaps_rhand_by_threshold.keys():
                print(
                    f"Percentage of frames with no right-hand tracking for threshold {threshold}: {sum([g for g in all_gaps_rhand_by_threshold[threshold] if g > 0]) / total_frames * 100:.2f}%"
                )

            print(f"Total frames in {phase}: {total_frames}")
            print(
                f"Total subjects in {phase}: {len(set(all_subjects))} - {all_subjects}"
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
