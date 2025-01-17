# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import json
import os
import random

from data_loaders.dataloader import load_data_from_manifold

from loguru import logger
from model.maskers import compute_masked_segments
from utils.config import pathmgr
from utils.constants import ConditionMasker, DatasetType


def main(
    dataset_name: str,
    dataset_path: str,
    store_path: str,
    eval_name: str,
    seed: int,
    masker: ConditionMasker,
    min_f: int,
    max_f: int,
    prob: float,
    min_dist: int,
    left_padding: int,
):
    """
    This function generates the gaps for the evaluation of the tracking loss recovery
    It stores the gaps in a json file to be later used by the evaluation script
    """
    json_path = os.path.join(store_path, eval_name + ".json")
    if pathmgr.exists(json_path):
        logger.info(f"Aborting... file already exists: {json_path}")
        return

    num_entities = len(ConditionMasker.get_entities_idces(masker, dataset_name))
    dataset_data = load_data_from_manifold(
        dataset_name,
        dataset_path,
        "test",
    )
    all_info = dataset_data.data
    filename_list = dataset_data.filename_list
    random.seed(seed)
    all_gaps = {}
    for idx, info in enumerate(all_info):
        seq_len = info["hmd_position_global_full_gt_list"].shape[0]
        filename = filename_list[idx]
        # use same maskers than for training to 1) reuse code and 2) align training <-> test as much as possible
        all_gaps[filename] = [
            compute_masked_segments(
                seq_len, prob, min_dist, min_f, max_f, left_padding=left_padding
            )
            for i in range(num_entities)
        ]
    # store to manifold
    metadata = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "eval_name": eval_name,
        "seed": seed,
        "masker": masker,
        "min_f": min_f,
        "max_f": max_f,
        "prob": prob,
        "min_dist": min_dist,
        "left_padding": left_padding,
    }
    pathmgr.mkdirs(store_path)
    with pathmgr.open(json_path, "w") as f:
        # save json
        json.dump(
            {
                "metadata": metadata,
                "gaps": all_gaps,
            },
            f,
            indent=4,
        )
    logger.info(f"Saved gaps to {store_path}")


def run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default=DatasetType.AMASS,
        type=DatasetType,
        help="Dataset name.",
    )
    parser.add_argument(
        "--dataset_path",
        default="manifold://xr_body/tree/personal/gbarquero/datasets/agrol/AMASS/new_format_data",
        type=str,
        help="Dataset path",
    )
    parser.add_argument(
        "--save_path",
        default="manifold://xr_body/tree/personal/gbarquero/datasets/agrol/AMASS/eval_gap_configs",
        type=str,
        help="Save path",
    )
    parser.add_argument(
        "--eval_name",
        type=str,
        help="Name for the evaluation",
    )
    parser.add_argument(
        "--seed",
        default=6,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--masker",
        default=ConditionMasker.SEQ_ALL,
        type=str,
        help="Type of masker for tracking loss segments generation (e.g., default, independent)",
    )
    parser.add_argument(
        "--min_f",
        default=1,
        type=int,
        help="Minimum frames inside masked segments.",
    )
    parser.add_argument(
        "--max_f",
        default=1,
        type=int,
        help="Maximum frames inside masked segments.",
    )
    parser.add_argument(
        "--min_dist",
        default=1,
        type=int,
        help="Minimum distance between two consecutive masked segments.",
    )
    parser.add_argument(
        "--left_padding",
        default=1,
        type=int,
        help="Number of frames at the beginning of the sequence that are not masked.",
    )
    parser.add_argument(
        "--prob",
        default=0.1,
        type=float,
        help="Probability of each frame starting a new masked segment.",
    )
    args = parser.parse_args()

    main(
        args.dataset,
        args.dataset_path,
        args.save_path,
        args.eval_name,
        args.seed,
        args.masker,
        args.min_f,
        args.max_f,
        args.prob,
        args.min_dist,
        args.left_padding,
    )


if __name__ == "__main__":
    run()
