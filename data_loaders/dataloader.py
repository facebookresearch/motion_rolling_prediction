# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import glob
import json
import multiprocessing as mp
import os
import sys
import tempfile

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
import utils.constants as constants
from loguru import logger

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.config import pathmgr
from utils.constants import (
    ConditionMasker,
    DatasetType,
    DataTypeGT,
    SMPLGenderParam,
    SMPLModelType,
)


@dataclass
class DatasetDataStruct:
    filename_list: List[str]
    data: List[dict]
    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class TrackingSignalGapsInfo:
    gaps: List[
        List[Tuple[int, int]]
    ]  # list with as many sublists of tuples (t0, t) as there are entities
    entities_idces: list  # list of idces of the entities in the sparse signal
    entities_smpl_idces: list  # list of idces of the entities in the SMPL model


def parse_data_struct(
    data_dict,
    keys_to_parse: Set[DataTypeGT],
    use_real_input=False,
    input_conf_threshold=0.0,
):
    """
    Parse the data structure to get the gt_dict and cond_dict used
    in training and inference.
    """
    gt_dict = {}
    cond_dict = {}
    if DataTypeGT.RELATIVE_ROTS in keys_to_parse:
        gt_dict[DataTypeGT.RELATIVE_ROTS] = data_dict["rotation_local_full_gt_list"]
    if DataTypeGT.SPARSE in keys_to_parse and not use_real_input:
        cond_dict[DataTypeGT.SPARSE] = data_dict["hmd_position_global_full_gt_list"]
    elif DataTypeGT.SPARSE in keys_to_parse:
        # use real input
        assert "hmd_position_global_full_real_list" in data_dict, "No real input found"
        assert "left_hand_confidence" in data_dict, "No left-hand-conf found"
        assert "right_hand_confidence" in data_dict, "No right-hand-conf found"
        sparse = data_dict["hmd_position_global_full_real_list"]  # [seq_len, 54]
        lhand_idces = constants.ENTITIES_IDCES[DatasetType.GORP][1]  # [seq_len]
        rhand_idces = constants.ENTITIES_IDCES[DatasetType.GORP][2]  # [seq_len]
        left_mask = data_dict["left_hand_confidence"] >= input_conf_threshold
        right_mask = data_dict["right_hand_confidence"] >= input_conf_threshold
        sparse[:, lhand_idces] = sparse[:, lhand_idces] * left_mask[:, None]
        sparse[:, rhand_idces] = sparse[:, rhand_idces] * right_mask[:, None]
        cond_dict[DataTypeGT.SPARSE] = sparse

    if DataTypeGT.GLOBAL_ROTS and "rotation_global_full_gt_list" in data_dict:
        gt_dict[DataTypeGT.GLOBAL_ROTS] = data_dict["rotation_global_full_gt_list"]
    if (
        DataTypeGT.WORLD_JOINTS in keys_to_parse
        and "position_global_full_gt_world" in data_dict
    ):
        gt_dict[DataTypeGT.WORLD_JOINTS] = data_dict["position_global_full_gt_world"]
    if DataTypeGT.BODY_PARAMS in keys_to_parse and "body_parms_list" in data_dict:
        gt_dict[DataTypeGT.BODY_PARAMS] = data_dict["body_parms_list"]
    if (
        DataTypeGT.HEAD_MOTION in keys_to_parse
        and "head_global_trans_list" in data_dict
    ):
        gt_dict[DataTypeGT.HEAD_MOTION] = data_dict["head_global_trans_list"]
    if (
        DataTypeGT.SHAPE_PARAMS in keys_to_parse
        and "body_parms_list" in data_dict
        and "betas" in data_dict["body_parms_list"]
    ):
        gt_dict[DataTypeGT.SHAPE_PARAMS] = data_dict["body_parms_list"]["betas"][1:]
    if DataTypeGT.SMPL_GENDER in keys_to_parse and "gender" in data_dict:
        gt_dict[DataTypeGT.SMPL_GENDER] = SMPLGenderParam[
            str(data_dict["gender"]).upper()
        ]
    if (
        DataTypeGT.SMPL_MODEL_TYPE in keys_to_parse
        and "surface_model_type" in data_dict
    ):
        model_type = data_dict["surface_model_type"]
        if not isinstance(model_type, str):
            model_type = model_type.item()
        gt_dict[DataTypeGT.SMPL_MODEL_TYPE] = SMPLModelType.parse(model_type)
    elif DataTypeGT.SMPL_MODEL_TYPE in keys_to_parse:
        gt_dict[DataTypeGT.SMPL_MODEL_TYPE] = SMPLModelType.SMPLH  # default

    if DataTypeGT.SHAPE_PARAMS not in gt_dict:
        # if shape params are not available, use the default shape --> retrocompatibility
        gt_dict[DataTypeGT.SMPL_GENDER] = SMPLGenderParam.MALE
    return gt_dict, cond_dict


class TrainDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        dataset_data: DatasetDataStruct,
        input_motion_length: int = 196,
        train_dataset_repeat_times: int = 1,
        no_normalization: bool = False,
        use_real_input: bool = False,
        input_conf_threshold: float = 0.0,
        **kwargs,
    ):
        self.dataset = dataset
        self.data = dataset_data.data
        self.mean = dataset_data.mean
        self.std = dataset_data.std
        self.std[abs(self.std) < 1e-6] = (
            1.0  # if std is 0 --> set it to 1 (value will be set to 0 after normalization)
        )
        self.mean_gpu = dataset_data.mean.to("cuda:0")
        self.std_gpu = dataset_data.std.to("cuda:0")
        self.filename_list = dataset_data.filename_list
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization
        self.input_motion_length = input_motion_length
        self.use_real_input = use_real_input
        self.input_conf_threshold = input_conf_threshold

    def __len__(self):
        return len(self.data) * self.train_dataset_repeat_times

    def transform(self, data):
        if self.no_normalization:
            return data
        return (data - self.mean) / (self.std + 1e-8)

    def inv_transform(self, data):
        if self.no_normalization:
            return data
        if data.is_cuda:
            return data * self.std_gpu + self.mean_gpu
        return data * self.std + self.mean

    def __getitem__(self, idx):
        data_dict = self.data[idx % len(self.data)]
        keys = {
            DataTypeGT.RELATIVE_ROTS,
            DataTypeGT.SPARSE,
            DataTypeGT.GLOBAL_ROTS,
            DataTypeGT.WORLD_JOINTS,
            DataTypeGT.SHAPE_PARAMS,
            DataTypeGT.SMPL_MODEL_TYPE,
            DataTypeGT.SMPL_GENDER,
        }
        gt_dict, cond_dict = parse_data_struct(
            data_dict, keys, self.use_real_input, self.input_conf_threshold
        )

        seqlen = cond_dict[DataTypeGT.SPARSE].shape[0]
        if seqlen <= self.input_motion_length:
            idx = 0
        else:
            idx = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]

        # slice the data
        for k in gt_dict:
            if isinstance(gt_dict[k], torch.Tensor):
                gt_dict[k] = gt_dict[k][idx : idx + self.input_motion_length].float()
        for k in cond_dict:
            if isinstance(cond_dict[k], torch.Tensor):
                cond_dict[k] = cond_dict[k][
                    idx : idx + self.input_motion_length
                ].float()

        # Normalization
        if not self.no_normalization:
            gt_dict[DataTypeGT.RELATIVE_ROTS] = self.transform(
                gt_dict[DataTypeGT.RELATIVE_ROTS]
            )

        return gt_dict, cond_dict


class OnlineTrainDataset(TrainDataset):
    def __init__(
        self,
        dataset: str,
        dataset_data: DatasetDataStruct,
        input_motion_length: int = 196,
        train_dataset_repeat_times: int = 1,
        no_normalization: bool = False,
        sparse_context: int = 0,
        motion_context: int = 0,
        freerunning_frames: int = 0,
        latency: int = 0,  # access tu future 'latency' frames of sparse info
        use_real_input: bool = False,
        input_conf_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            dataset,
            dataset_data,
            input_motion_length,
            train_dataset_repeat_times,
            no_normalization,
            use_real_input,
            input_conf_threshold,
        )
        self.latency = latency
        self.freerunning_frames = freerunning_frames
        self.sparse_ctx_len = sparse_context
        self.motion_ctx_len = motion_context
        self.pred_len = input_motion_length
        self.total_len = (
            input_motion_length
            + max(sparse_context, motion_context)
            + freerunning_frames
        )

    def __getitem__(self, idx):
        data_dict = self.data[idx % len(self.data)]
        keys = {
            DataTypeGT.RELATIVE_ROTS,
            DataTypeGT.SPARSE,
            DataTypeGT.GLOBAL_ROTS,
            DataTypeGT.WORLD_JOINTS,
            DataTypeGT.SHAPE_PARAMS,
            DataTypeGT.SMPL_MODEL_TYPE,
            DataTypeGT.SMPL_GENDER,
        }
        gt_dict, cond_dict = parse_data_struct(
            data_dict, keys, self.use_real_input, self.input_conf_threshold
        )

        seqlen = cond_dict[DataTypeGT.SPARSE].shape[0]
        if seqlen <= self.total_len:
            idx = 0
        else:
            idx = torch.randint(0, int(seqlen - self.total_len), (1,))[0].item()

        # slice the data
        pred_idx0 = idx + max(self.sparse_ctx_len, self.motion_ctx_len)
        cond_dict[DataTypeGT.MOTION_CTX] = gt_dict[DataTypeGT.RELATIVE_ROTS][
            pred_idx0 - self.motion_ctx_len : pred_idx0 + self.freerunning_frames
        ].float()
        cond_dict[DataTypeGT.SPARSE] = cond_dict[DataTypeGT.SPARSE][
            pred_idx0
            - self.sparse_ctx_len : pred_idx0
            + 1
            + self.latency
            + self.freerunning_frames
        ].float()  # online --> self.context in the past and current frame

        for k in gt_dict:
            if isinstance(gt_dict[k], torch.Tensor):
                gt_dict[k] = gt_dict[k][
                    pred_idx0 : pred_idx0 + self.pred_len + self.freerunning_frames
                ]

        # Normalization
        if not self.no_normalization:
            gt_dict[DataTypeGT.RELATIVE_ROTS] = self.transform(
                gt_dict[DataTypeGT.RELATIVE_ROTS]
            )
            cond_dict[DataTypeGT.MOTION_CTX] = self.transform(
                cond_dict[DataTypeGT.MOTION_CTX]
            )

        return gt_dict, cond_dict


class TestDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_path: str,
        no_normalization: bool,
        max_samples: int = -1,
        normalize_sparse: str = "none",
        min_frames: int = 0,
        max_frames: int = sys.maxsize,
        eval_gap_config: Optional[str] = None,
        num_features: Optional[int] = None,
        use_real_input: bool = False,
        input_conf_threshold: float = 0.0,
        test_split: str = "test",
        **kwargs,
    ):
        dataset_data = load_data_from_manifold(
            name,
            dataset_path,
            test_split,
            max_samples=max_samples,
        )
        self.name = name
        self.mean = dataset_data.mean
        self.std = dataset_data.std
        self.normalize_sparse = normalize_sparse
        self.no_normalization = no_normalization
        self.use_real_input = use_real_input
        self.input_conf_threshold = input_conf_threshold

        self.filename_list = []
        self.data = []
        filtered = 0
        for filename, info in zip(dataset_data.filename_list, dataset_data.data):
            hmd = info["hmd_position_global_full_gt_list"]
            if hmd.shape[0] < min_frames or hmd.shape[0] > max_frames:
                filtered += 1
                continue
            if info["rotation_local_full_gt_list"] is None:
                assert (
                    num_features is not None
                ), "num_features must be provided when GT is not available"
                info["rotation_local_full_gt_list"] = torch.zeros(
                    (hmd.shape[0], num_features)
                )
            self.data.append(info)
            self.filename_list.append(filename)
        logger.info(f"Filtered {filtered}/{len(dataset_data.data)} sequences")

        self.eval_gap_config = eval_gap_config
        if eval_gap_config is not None:
            self.tracking_gaps = self.inject_eval_gaps(name, dataset_path)
            logger.info("Tracking gaps applied to sparse tracking signal!")
        else:
            self.tracking_gaps = None

    def inject_eval_gaps(
        self, dataset_name: str, dataset_path: str
    ) -> List[TrackingSignalGapsInfo]:
        assert self.eval_gap_config is not None, "No eval gap config provided!"
        gaps_json = load_gaps_from_manifold(
            dataset_name, dataset_path, self.eval_gap_config
        )
        # sparse shape --> [seq_len, feats]
        masker_type = ConditionMasker.parse(gaps_json["metadata"]["masker"])
        entities_idces = ConditionMasker.get_entities_idces(masker_type, dataset_name)
        entities_smpl_idces = ConditionMasker.get_entities_smpl_idces(
            masker_type, dataset_name
        )
        all_gaps = []
        for i, filename in enumerate(self.filename_list):
            assert filename in gaps_json["gaps"], f"{filename} not in {gaps_json}"
            entities_gaps = gaps_json["gaps"][filename]
            assert len(entities_gaps) == len(entities_idces)
            for entity_idx, gaps in enumerate(entities_gaps):
                for t0, t in gaps:
                    self.data[i]["hmd_position_global_full_gt_list"][
                        t0:t, entities_idces[entity_idx]
                    ] = 0
            all_gaps.append(
                TrackingSignalGapsInfo(
                    entities_gaps, entities_idces, entities_smpl_idces
                )
            )
        return all_gaps

    def __len__(self):
        return len(self.data)

    def transform(self, data):
        if self.no_normalization:
            return data
        return (data - self.mean) / (self.std + 1e-8)

    def inv_transform(self, data):
        if self.no_normalization:
            return data
        return data * self.std + self.mean

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        keys = {
            DataTypeGT.RELATIVE_ROTS,
            DataTypeGT.SPARSE,
            DataTypeGT.GLOBAL_ROTS,
            DataTypeGT.WORLD_JOINTS,
            DataTypeGT.SHAPE_PARAMS,
            DataTypeGT.BODY_PARAMS,
            DataTypeGT.HEAD_MOTION,
            DataTypeGT.SMPL_MODEL_TYPE,
            DataTypeGT.SMPL_GENDER,
        }
        gt_dict, cond_dict = parse_data_struct(
            data_dict, keys, self.use_real_input, self.input_conf_threshold
        )
        gt_dict[DataTypeGT.TRACKING_GAP] = (
            self.tracking_gaps[idx] if self.tracking_gaps else None
        )
        gt_dict[DataTypeGT.FILENAME] = self.filename_list[idx]
        gt_dict[DataTypeGT.NUM_FRAMES] = data_dict[
            "hmd_position_global_full_gt_list"
        ].shape[0]
        return gt_dict, cond_dict


def get_mean_std_path(dataset):
    return dataset + "_mean.pt", dataset + "_std.pt"


def get_motion(motion_list):
    # rotation_local_full_gt_list : 6d rotation parameters
    # hmd_position_global_full_gt_list : 3 joints(head, hands) 6d rotation/6d rotation velocity/global translation/global translation velocity
    motions = [i["rotation_local_full_gt_list"] for i in motion_list]
    sparses = [i["hmd_position_global_full_gt_list"] for i in motion_list]
    return motions, sparses


def get_path(dataset_path, split):
    data_list_path = []
    parent_data_path = glob.glob(dataset_path + "/*")
    for d in parent_data_path:
        if os.path.isdir(d):
            files = glob.glob(d + "/" + split + "/*pt")
            data_list_path.extend(files)
    return data_list_path


def get_manifold_paths(dataset_path, split):
    data_list_path = []
    parent_data_path = pathmgr.ls(dataset_path)
    for d in parent_data_path:
        if pathmgr.isdir(os.path.join(dataset_path, d, split)):
            files = [
                os.path.join(dataset_path, d, split, f)
                for f in pathmgr.ls(os.path.join(dataset_path, d, split))
                if f.endswith(".pt")
            ]
            data_list_path.extend(files)
    return data_list_path


def load_data_from_manifold(
    dataset,
    dataset_path,
    split,
    total_length=196,
    max_samples=-1,
) -> DatasetDataStruct:
    """
    Collect the data for the given split

    Args:
        - For test:
            dataset : the name of the testing dataset
            split : test or train
        - For train:
            dataset : the name of the training dataset
            split : train or test
            input_motion_length : the input motion length

    Outout:
        - For test:
            filename_list : List of all filenames in the dataset
            motion_list : List contains N dictoinaries, with
                        "hmd_position_global_full_gt_list" - sparse features of the 3 joints
                        "local_joint_parameters_gt_list" - body parameters Nx7[tx,ty,tz,rx,ry,rz] as the input of the human kinematic model
                        "head_global_trans_list" - Tx4x4 matrix which contains the global rotation and global translation of the head movement
            mean : mean of train dataset
            std : std of train dataset
        - For train:
            new_motions : motions indicates the sequences of rotation representation of each joint
            new_sparses : sparses indicates the sequences of sparse features of the 3 joints
            mean : mean of train dataset
            std : std of train dataset
    """

    motion_list = get_manifold_paths(dataset_path, split)
    if max_samples != -1:
        motion_list = motion_list[:max_samples]
    mean_path, std_path = get_mean_std_path(dataset)
    logger.info(f"Loading '{split}' data from manifold: {dataset_path}")
    num_mp = 8 if split != "train" else 16
    if len(motion_list) < 10:
        num_mp = 1
    with mp.Pool(num_mp) as p:
        local_paths = list(
            tqdm(p.imap(pathmgr.get_local_path, motion_list), total=len(motion_list))
        )
        data = [torch.load(i) for i in tqdm(local_paths)]
        p.close()
        p.join()

    filename_list = [
        "-".join([i.split("/")[-3], i.split("/")[-1]]).split(".")[0]
        for i in motion_list
    ]
    if "test" in split:
        mean = torch.load(
            pathmgr.get_local_path(os.path.join(dataset_path, mean_path)),
            weights_only=True,
        )
        std = torch.load(
            pathmgr.get_local_path(os.path.join(dataset_path, std_path)),
            weights_only=True,
        )
        return DatasetDataStruct(filename_list, data, mean, std)

    assert split == "train"

    print("Filtering training sequences")
    filtered_data = []
    filtered_filenames = []
    for idx, data_dict in enumerate(data):
        nframes = data_dict["hmd_position_global_full_gt_list"].shape[0]
        if nframes < total_length:
            continue
        filtered_data.append(data_dict)
        filtered_filenames.append(filename_list[idx])
    print(f"Before filtering: {len(data)}\n  After filtering: {len(filtered_data)}")

    if pathmgr.exists(os.path.join(dataset_path, mean_path)):
        mean = torch.load(pathmgr.get_local_path(os.path.join(dataset_path, mean_path)))
        std = torch.load(pathmgr.get_local_path(os.path.join(dataset_path, std_path)))
        print("Loading mean and std from manifold")
    else:
        print(
            "Mean and std not available in manifold! Computing mean and std from training data..."
        )
        all_motions = [d["rotation_local_full_gt_list"] for d in filtered_data]
        tmp_data_list = torch.cat(all_motions, dim=0)
        mean = tmp_data_list.mean(dim=0).float()
        std = tmp_data_list.std(dim=0).float()
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(os.path.join(tmpdirname, mean_path), "wb") as f:
                torch.save(mean, f)
            with open(os.path.join(tmpdirname, std_path), "wb") as f:
                torch.save(std, f)
            pathmgr.copy(
                src_path=os.path.join(tmpdirname, mean_path),
                dst_path=os.path.join(dataset_path, mean_path),
                overwrite=True,
            )
            pathmgr.copy(
                src_path=os.path.join(tmpdirname, std_path),
                dst_path=os.path.join(dataset_path, std_path),
                overwrite=True,
            )
        print("Mean and std saved to manifold: ", dataset_path)

    return DatasetDataStruct(filtered_filenames, filtered_data, mean, std)


def load_gaps_from_manifold(
    dataset: str,
    dataset_path: str,
    eval_name: str,
):
    folder_path = os.path.dirname(dataset_path)
    manifold_path = os.path.join(folder_path, "eval_gap_configs", eval_name + ".json")
    assert pathmgr.exists(manifold_path), f"File {manifold_path} does not exist"
    local_path = pathmgr.get_local_path(manifold_path)
    with open(local_path, "r") as f:
        data = json.load(f)
    return data


def load_data(dataset, dataset_path, split, **kwargs):
    """
    Collect the data for the given split

    Args:
        - For test:
            dataset : the name of the testing dataset
            split : test or train
        - For train:
            dataset : the name of the training dataset
            split : train or test
            input_motion_length : the input motion length

    Outout:
        - For test:
            filename_list : List of all filenames in the dataset
            motion_list : List contains N dictoinaries, with
                        "hmd_position_global_full_gt_list" - sparse features of the 3 joints
                        "local_joint_parameters_gt_list" - body parameters Nx7[tx,ty,tz,rx,ry,rz] as the input of the human kinematic model
                        "head_global_trans_list" - Tx4x4 matrix which contains the global rotation and global translation of the head movement
            mean : mean of train dataset
            std : std of train dataset
        - For train:
            new_motions : motions indicates the sequences of rotation representation of each joint
            new_sparses : sparses indicates the sequences of sparse features of the 3 joints
            mean : mean of train dataset
            std : std of train dataset
    """

    if split == "test":
        motion_list = get_path(dataset_path, split)
        mean_path, std_path = get_mean_std_path(dataset)
        filename_list = [
            "-".join([i.split("/")[-3], i.split("/")[-1]]).split(".")[0]
            for i in motion_list
        ]
        motion_list = [torch.load(i) for i in tqdm(motion_list)]
        mean = torch.load(os.path.join(dataset_path, mean_path))
        std = torch.load(os.path.join(dataset_path, std_path))
        return filename_list, motion_list, mean, std

    assert split == "train"
    assert (
        "input_motion_length" in kwargs
    ), "Please specify the input_motion_length to load training dataset"

    motion_list = get_path(dataset_path, split)
    mean_path, std_path = get_mean_std_path(dataset)
    input_motion_length = kwargs["input_motion_length"]
    motion_list = [torch.load(i) for i in tqdm(motion_list)]

    motions, sparses = get_motion(motion_list)

    new_motions = []
    new_sparses = []
    for idx, motion in enumerate(motions):
        if motion.shape[0] < input_motion_length:  # Arbitrary choice
            continue
        new_sparses.append(sparses[idx])
        new_motions.append(motions[idx])

    if os.path.exists(os.path.join(dataset_path, mean_path)):
        mean = torch.load(os.path.join(dataset_path, mean_path))
        std = torch.load(os.path.join(dataset_path, std_path))
    else:
        tmp_data_list = torch.cat(new_motions, dim=0)
        mean = tmp_data_list.mean(axis=0).float()
        std = tmp_data_list.std(axis=0).float()
        with open(os.path.join(dataset_path, mean_path), "wb") as f:
            torch.save(mean, f)
        with open(os.path.join(dataset_path, std_path), "wb") as f:
            torch.save(std, f)

    return new_motions, new_sparses, mean, std


def get_dataloader(dataset, split, batch_size, num_workers=32, persistent_workers=True):

    if split == "train":
        shuffle = True
        drop_last = True
        num_workers = num_workers
    else:
        shuffle = False
        drop_last = False
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    return loader
