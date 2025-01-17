import os
import sys

import torch

sys.path.append(os.path.dirname(__file__))
import utils.joints2smpl.config as config
from utils.joints2smpl.customloss import body_fitting_loss_3d, camera_fitting_loss_3d
from utils.joints2smpl.prior import MaxMixturePrior


@torch.no_grad()
def guess_init_3d(model_joints, j3d, joints_category="orig"):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model_joints: SMPL model with pre joints
    :param j3d: 25x3 array of Kinect Joints
    :returns: 3D vector corresponding to the estimated camera translation
    """
    # get the indexed four
    gt_joints = ["RHip", "LHip", "RShoulder", "LShoulder"]
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]

    if joints_category == "orig":
        joints_ind_category = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category == "AMASS":
        joints_ind_category = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        print("NO SUCH JOINTS CATEGORY!")

    sum_init_t = (j3d[:, joints_ind_category] - model_joints[:, gt_joints_ind]).sum(
        dim=1
    )
    init_t = sum_init_t / 4.0
    return init_t


# SMPLIfy 3D
class SMPLify3D:
    """Implementation of SMPLify, use 3D joints."""

    def __init__(
        self,
        smplxmodel,
        step_size=1e-2,
        num_iters=100,
        use_lbfgs=True,
        joints_category="orig",
        device=torch.device("cuda:0"),
    ):

        # Store options
        self.device = device
        self.step_size = step_size

        self.STEP_1_REPS = 5  # fit global orientation
        self.STEP_2_REPS = 10  # fit body pose and global orientation
        self.num_iters = num_iters
        # --- choose optimizer
        self.use_lbfgs = use_lbfgs
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(
            prior_folder=config.GMM_MODEL_DIR, num_gaussians=8, dtype=torch.float32
        ).to(device)

        # reLoad SMPL-X model
        self.smpl = smplxmodel

        # select joint joint_category
        self.joints_category = joints_category

        if joints_category == "orig":
            self.smpl_index = config.full_smpl_idx
            self.corr_index = config.full_smpl_idx
        elif joints_category == "AMASS":
            self.smpl_index = config.amass_smpl_idx
            self.corr_index = config.amass_idx
        else:
            self.smpl_index = None
            self.corr_index = None
            print("NO SUCH JOINTS CATEGORY!")

    # ---- get the man function here ------
    def __call__(self, init_pose, init_betas, j3d, conf_3d=1.0):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            j3d: joints 3d aka keypoints
            conf_3d: confidence for 3d joints
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
        """
        idces_to_nullify = [27, 28, 29, 30, 31, 32]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # use guess 3d to get the initial
        smpl_output = self.smpl(
            global_orient=global_orient, body_pose=body_pose, betas=betas
        )
        model_joints = smpl_output.Jtr[:, : j3d.shape[1], :]

        preserve_pose = init_pose[:, 3:].detach().clone()
        # -------------Step 1: Optimize body orientation--------
        # logger.info("[1] Optimizing body orientation...")
        # Optimize only body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True

        camera_opt_params = [global_orient]

        if self.use_lbfgs:
            camera_optimizer = torch.optim.LBFGS(
                camera_opt_params,
                max_iter=self.num_iters,
                lr=self.step_size,
                line_search_fn="strong_wolfe",
            )
            for _ in range(self.STEP_1_REPS):

                def closure():
                    camera_optimizer.zero_grad()
                    smpl_output = self.smpl(
                        global_orient=global_orient, body_pose=body_pose, betas=betas
                    )
                    model_joints = smpl_output.Jtr[:, : j3d.shape[1], :]

                    loss = camera_fitting_loss_3d(
                        model_joints,
                        j3d,
                        self.joints_category,
                    )
                    loss.backward()
                    return loss

                camera_optimizer.step(closure)
        else:
            camera_optimizer = torch.optim.Adam(
                camera_opt_params, lr=self.step_size, betas=(0.9, 0.999)
            )

            for _ in range(self.STEP_1_REPS * 2):
                smpl_output = self.smpl(
                    global_orient=global_orient, body_pose=body_pose, betas=betas
                )
                model_joints = smpl_output.Jtr[:, : j3d.shape[1], :]

                loss = camera_fitting_loss_3d(
                    model_joints[:, self.smpl_index],
                    j3d[:, self.corr_index],
                    self.joints_category,
                )
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()

        # Fix camera translation after optimizing camera
        # --------Step 2: Optimize body joints --------------------------
        # logger.info("[2] Optimizing body joints...")
        # Optimize only the body pose and global orientation of the body
        body_pose[:, idces_to_nullify] = 0
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        betas.requires_grad = False
        body_opt_params = [body_pose, global_orient]

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(
                body_opt_params,
                max_iter=self.num_iters,
                lr=self.step_size,
                line_search_fn="strong_wolfe",
            )
            for _ in range(self.STEP_2_REPS):

                def closure():
                    body_optimizer.zero_grad()
                    smpl_output = self.smpl(
                        global_orient=global_orient, body_pose=body_pose, betas=betas
                    )
                    model_joints = smpl_output.Jtr[:, : j3d.shape[1], :]

                    loss = body_fitting_loss_3d(
                        body_pose,
                        preserve_pose,
                        model_joints[:, self.smpl_index],
                        j3d[:, self.corr_index],
                        self.pose_prior,
                        joints3d_conf=conf_3d,
                        joint_loss_weight=600.0,
                        pose_preserve_weight=5.0,
                    )
                    loss.backward()
                    body_pose.grad[:, idces_to_nullify] = 0
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(
                body_opt_params, lr=self.step_size, betas=(0.9, 0.999)
            )

            for _ in range(self.STEP_2_REPS):
                smpl_output = self.smpl(
                    global_orient=global_orient, body_pose=body_pose, betas=betas
                )
                model_joints = smpl_output.Jtr[:, : j3d.shape[1], :]

                loss = body_fitting_loss_3d(
                    body_pose,
                    preserve_pose,
                    model_joints[:, self.smpl_index],
                    j3d[:, self.corr_index],
                    self.pose_prior,
                    joints3d_conf=conf_3d,
                    joint_loss_weight=600.0,
                )
                body_optimizer.zero_grad()
                loss.backward()
                body_pose.grad[:, idces_to_nullify] = 0
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
            )
            model_joints = smpl_output.Jtr[:, : j3d.shape[1], :]

            final_loss = body_fitting_loss_3d(
                body_pose,
                preserve_pose,
                model_joints[:, self.smpl_index],
                j3d[:, self.corr_index],
                self.pose_prior,
                joints3d_conf=conf_3d,
                joint_loss_weight=600.0,
            )

        vertices = smpl_output.v.detach()
        joints = smpl_output.Jtr[:, : j3d.shape[1], :].detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, final_loss
