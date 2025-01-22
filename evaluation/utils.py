# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
from typing import Optional

import torch

from human_body_prior.body_model.body_model import BodyModel as BM
from loguru import logger

from utils import utils_transform

from utils.constants import DatasetType, SMPLGenderParam, SMPLModelType

DATASETS_FPS = {
    DatasetType.AMASS_P1: 60,
    DatasetType.AMASS_P2: 30,
    DatasetType.GORP: 30,
}

#  1 second of left padding in evaluation for all datasets
MIN_FRAMES_TO_EVAL = {k: fps * 1 for k, fps in DATASETS_FPS.items()}

#####################


class BodyModel(torch.nn.Module):
    def __init__(
        self,
        support_dir,
        device,
        smpl_gender: SMPLGenderParam,
        model_type: SMPLModelType,
    ):
        super().__init__()
        smpl_gender = smpl_gender.value.lower()
        bm_fname = os.path.join(
                support_dir, model_type.lower(), smpl_gender.lower(),
                "model.npz",
            )
        num_betas = 16  # number of body parameters
        if model_type == SMPLModelType.SMPLH:
            dmpl_fname = os.path.join(support_dir, "dmpls", smpl_gender.lower(), "model.npz")
            num_dmpls = 8  # number of DMPL parameters
        else:
            dmpl_fname = None
            num_dmpls = None
        body_model = BM(
            bm_fname=bm_fname,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        )
        self.body_model = body_model.to(device)
        self.device = device

    def get_smplify_fn(self):
        def forward(global_orient, body_pose, betas):
            return self.body_model(
                betas=betas,
                root_orient=global_orient,
                pose_body=body_pose,
            )

        return forward

    def to(self, device):
        self.device = device
        self.body_model.to(device)
        return self

    def forward(self, body_params: dict):
        self.body_model.eval()
        with torch.no_grad():
            body_pose = self.body_model(**body_params)
        return body_pose

    def grad_fk(self, body_params: dict):
        self.body_model.train()
        return self.body_model(**body_params)


class BodyModelsWrapper(torch.nn.Module):
    def __init__(self, support_dir: str):
        super().__init__()
        self.support_dir = support_dir
        self.body_models = {}

    def get_body_model_key(self, model_type: SMPLModelType, gender: SMPLGenderParam):
        return f"{model_type.value}/{gender.value}"

    def get_body_model(
        self,
        model_type: SMPLModelType,
        gender: SMPLGenderParam,
        device: Optional[str] = None,
    ):
        key = self.get_body_model_key(model_type, gender)
        if key not in self.body_models:
            logger.info(f"Loading body model {key}")
            self.body_models[key] = BodyModel(
                self.support_dir,
                device if device is not None else "cpu",
                gender,
                model_type=model_type,
            )
        elif device is not None and self.body_models[key].device != device:
            self.body_models[key].to(device)
        return self.body_models[key]

    def get_device_From_params(self, body_params: dict):
        assert len(body_params.keys()) > 0, "body_params is empty"
        device = body_params[list(body_params.keys())[0]].device
        return device

    def to(self, device):
        for key in self.body_models:
            self.body_models[key].to(device)
        return self

    def get_kin_tree(self, model_type: SMPLModelType, gender: SMPLGenderParam):
        return self.get_body_model(model_type, gender).body_model.kintree_table

    def forward(
        self, body_params: dict, model_type: SMPLModelType, gender: SMPLGenderParam
    ):
        dev = self.get_device_From_params(body_params)
        bm = self.get_body_model(model_type, gender, dev)
        return bm(body_params)

    def grad_fk(
        self, body_params: dict, model_type: SMPLModelType, gender: SMPLGenderParam
    ):
        dev = self.get_device_From_params(body_params)
        bm = self.get_body_model(model_type, gender, dev)
        return bm.grad_fk(body_params)


def get_GT_body_poses(
    body_model: BodyModelsWrapper,
    body_param: dict,
    seq_len: int,
    device: str,
    gender: SMPLGenderParam,
    model_type: SMPLModelType,
):
    # ================ GT (plain transform through body model) ==================
    for k, v in body_param.items():
        body_param[k] = v.squeeze().to(device)
        body_param[k] = body_param[k][-seq_len:, ...]

    # Get the ground truth position from the model
    gt_body = body_model(body_param, model_type, gender)
    return gt_body


def get_body_poses(
    motion_pred: torch.Tensor,
    body_model: BodyModelsWrapper,
    head_motion: torch.Tensor,
    device: str,
    gender: SMPLGenderParam,
    model_type: SMPLModelType,
    betas: Optional[torch.Tensor] = None,
):

    # ================ Pred (integrate HMD global motion) ==================
    motion_pred = motion_pred.to(device)

    # Get the  prediction from the model
    model_rot_input = (
        utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach())
        .reshape(motion_pred.shape[0], -1)
        .float()
    )

    T_head2world = head_motion.clone().to(device)
    t_head2world = T_head2world[:, :3, 3].clone()

    # Get the offset between the head and other joints using forward kinematic model
    body_params = {
        "pose_body": model_rot_input[..., 3:66],
        "root_orient": model_rot_input[..., :3],
    }
    if betas is not None:
        betas = betas.to(device)
        body_params["betas"] = betas
    body_pose_local = body_model(body_params, model_type, gender).Jtr

    # Get the offset in global coordiante system between head and body_world.
    t_head2root = -body_pose_local[:, 15, :]
    t_root2world = t_head2root + t_head2world.to(device)

    body_params = {
        "pose_body": model_rot_input[..., 3:66],
        "root_orient": model_rot_input[..., :3],
        "trans": t_root2world,
    }
    if betas is not None:
        body_params["betas"] = betas
    predicted_body = body_model(body_params, model_type, gender)

    return predicted_body


def get_dataset_fps(dataset_name):
    if dataset_name in DATASETS_FPS:
        return DATASETS_FPS[dataset_name]
    raise ValueError(f"Dataset {dataset_name} not found in DATASETS_FPS.")


def get_min_frames_to_eval(dataset_name):
    if dataset_name in MIN_FRAMES_TO_EVAL:
        return MIN_FRAMES_TO_EVAL[dataset_name]
    raise ValueError(f"Dataset {dataset_name} not found in MIN_FRAMES_TO_EVAL.")
