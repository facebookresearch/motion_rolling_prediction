# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import os

from loguru import logger
from tqdm import tqdm

TRAIN_DATASETS = [
    "CMU",
    "MPI_Limits",
    "TotalCapture",
    "Eyes_Japan_Dataset",
    "KIT",
    "BioMotionLab_NTroje",
    "BMLmovi",
    "EKUT",
    "ACCAD",
    "MPI_HDM05",
    "SFU",
    "MPI_mosh",
]
TEST_DATASETS = ["Transitions_mocap", "HumanEva"]


def clean_splits(args, dataset_name, filepaths):
    """
    Replicated from https://github.com/davrempe/humor/blob/main/humor/scripts/cleanup_amass_data.py
    It removes the treadmill examples from BioMotionLab_NTroje and the Figure-Skating examples from MPI_HDM05.
    """
    if args.clean_splits:
        if dataset_name == "BioMotionLab_NTroje":
            return [f for f in filepaths if "treadmill" not in f and "normal" not in f]
        elif dataset_name == "MPI_HDM05":
            return [f for f in filepaths if "HDM_dg_07-01" not in f]
    return filepaths


def generate_filepaths(args):
    if args.use_vip:
        from iopath import PathManager
        from iopath.fb.manifold import ManifoldPathHandler

        pathmgr = PathManager()
        pathmgr.register_handler(ManifoldPathHandler(use_vip=True))
    else:
        from utils.config import pathmgr
    for phase, datasets_list in zip(["train", "test"], [TRAIN_DATASETS, TEST_DATASETS]):
        for dataset in datasets_list:
            split_folder = os.path.join(args.splits_dir, dataset)
            os.makedirs(split_folder, exist_ok=True)

            split_file = os.path.join(split_folder, phase + "_split.txt")
            # parse all the files in the dataset
            dataset_folder = os.path.join(args.root_dir, dataset)
            assert pathmgr.exists(dataset_folder), f"{dataset_folder} does not exist"
            subfolders = [
                os.path.join(dataset_folder, f)
                for f in pathmgr.ls(dataset_folder)
                if pathmgr.isdir(os.path.join(dataset_folder, f))
            ]
            # for each subfolder, get the filepaths
            filepaths = []
            for subfolder in tqdm(subfolders):
                filepaths += [
                    os.path.join(subfolder, file)
                    for file in pathmgr.ls(subfolder)
                    if file.endswith(".npz") and "poses" in file
                ]
            filepaths = clean_splits(args, dataset, filepaths)
            # write the filepaths to the split file
            assert len(filepaths) > 0, f"No files found in {dataset_folder}"
            # assert not os.path.exists(split_file), f"{split_file} already exists"
            with open(split_file, "w") as f:
                for filepath in filepaths:
                    # remove the root_folder
                    filepath = filepath.replace(args.root_dir, "")
                    # write the filepath to the file
                    f.write(filepath + "\n")
            logger.info(f"[{phase}] Saved {len(filepaths)} filepaths to {split_file}")


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
    parser.add_argument(
        "--use_vip",
        action="store_true",
        help="If True, it will use the --vip argument when accessing the Manifold bucket.",
    )
    parser.add_argument(
        "--clean_splits",
        action="store_true",
        help="If True, it will clean some files following https://github.com/davrempe/humor/blob/main/humor/scripts/cleanup_amass_data.py.",
    )
    args = parser.parse_args()

    generate_filepaths(args)


if __name__ == "__main__":
    run()
