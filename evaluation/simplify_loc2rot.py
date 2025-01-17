# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# adapted from https://github.com/GuyTevet/motion-diffusion-model/blob/main/visualize/simplify_loc2rot.py

import h5py
import torch
import utils.joints2smpl.config as config
from utils.joints2smpl.smplify import SMPLify3D


class joints2smpl:

    def __init__(self, smplmodel, device, num_betas=10, num_joints=22):
        self.device = device
        self.num_joints = num_joints
        self.num_betas = num_betas
        self.joint_category = "AMASS"
        self.num_smplify_iters = 20
        self.fix_foot = False

        # ## --- load the mean pose as original ----
        smpl_mean_file = config.SMPL_MEAN_FILE
        file = h5py.File(smpl_mean_file, "r")

        self.init_mean_pose = torch.from_numpy(file["pose"][:]).float().to(self.device)
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)
        #

        # # #-------------initialize SMPLify
        self.smplify = SMPLify3D(
            smplxmodel=smplmodel,
            joints_category=self.joint_category,
            num_iters=self.num_smplify_iters,
            device=self.device,
        )

    def joint2smpl(self, input_joints, confidence_input, betas, init_params=None):
        batch_size = input_joints.shape[0]
        _smplify = self.smplify  # if init_params is None else self.smplify_fast
        pred_pose = torch.zeros(batch_size, self.num_joints * 3).to(self.device)
        pred_betas = torch.zeros(batch_size, self.num_betas).to(self.device)
        keypoints_3d = torch.Tensor(input_joints).to(self.device).float()

        if init_params is None:
            assert betas is not None, "betas must be provided if init_params is None"
            n_betas = min(self.num_betas, betas.shape[1])
            pred_betas[:, :n_betas] = betas[:, :n_betas]
            n_pose_params = min(self.num_joints * 3, self.init_mean_pose.shape[0])
            pred_pose[:, :n_pose_params] = self.init_mean_pose[
                :n_pose_params
            ].unsqueeze(0)
        else:
            pred_betas = init_params["betas"]
            pred_pose = init_params["pose"]

        (
            new_opt_vertices,
            new_opt_joints,
            new_opt_pose,
            new_opt_betas,
            new_opt_joint_loss,
        ) = _smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(self.device),
        )
        return new_opt_pose, new_opt_joints
