# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import json
import os
import tempfile
from typing import List, Optional

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch as th
import utils.constants as constants
from data_loaders.dataloader import TrackingSignalGapsInfo
from evaluation.utils import (
    BodyModelsWrapper,
    get_body_poses,
    get_dataset_fps,
    get_GT_body_poses,
)

from loguru import logger
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from utils import utils_transform

from utils.config import pathmgr
from utils.constants import (
    DatasetType,
    DataTypeGT,
    ModelOutputType,
    RollingVisType,
    SMPLGenderParam,
    SMPLModelType,
)


VIS_SAMPLES_SUBSET = {
    DatasetType.AMASS: [
        "BioMotionLab_NTroje-110",
        "BioMotionLab_NTroje-58",
        "BioMotionLab_NTroje-13",
        "BioMotionLab_NTroje-138",
        "BioMotionLab_NTroje-125",
        "CMU-158",
        "MPI_HDM05-1",
        "CMU-156",
        "CMU-186",
        "CMU-173",
        "BioMotionLab_NTroje-4",
        "BioMotionLab_NTroje-89",
        "BioMotionLab_NTroje-299",
        "BioMotionLab_NTroje-291",
    ],
    DatasetType.AMASSFULL: [
        "Transitions_mocap-17",
        "Transitions_mocap-20",
        "Transitions_mocap-54",
        "Transitions_mocap-9",
        "HumanEva-24",
        "HumanEva-2",
    ],
    DatasetType.GORP: [
        "GORP-20",
        "GORP-40",
        "GORP-60",
        "GORP-80",
        "GORP-100",
        "GORP-150",
        "GORP-200",
        "GORP-250",
        "GORP-300",
    ],
    DatasetType.ITW: [
        "motion_controllers-1",
        # "motion_controllers-2",
        # "motion_controllers-3",
        # "motion_controllers-4",
        # "motion_controllers-5",
        # "motion_controllers-6",
        "hand_tracking-1",
        "hand_tracking-2",
        "hand_tracking-3",
        "hand_tracking-4",
        "hand_tracking-5",
        "hand_tracking-6",
        "hand_tracking-7",
        "hand_tracking-8",
        "hand_tracking-9",
        "hand_tracking-10",
    ],
}


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list) or isinstance(obj, tuple):
            # round to 4 decimal places
            return [round(x, 4) for x in obj]
        else:
            return super().default(obj)


def get_marker_points_and_colors(
    gaps: Optional[TrackingSignalGapsInfo], sparse, pr_body=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # gt_body, pr_body are of a type defined in https://fburl.com/code/rmb5rabq
    # RGBA
    GREEN = [0.0, 1.0, 0.0, 1.0]
    RED = [1.0, 0.0, 0.0, 1.0]
    GRAY_TR = [0.0, 0.0, 1.0, 0.8]
    SMPL_TO_POINT = {
        15: 0,
        20: 1,
        21: 2,
    }
    to_show = list(SMPL_TO_POINT.keys())
    # ===== GT extracted from SPARSE =====
    points_GT = (
        torch.stack(
            [
                sparse[:, 36:39],
                sparse[:, 39:42],
                sparse[:, 42:45],
            ],
            dim=1,
        )
        .cpu()
        .numpy()
    )
    rots_GT = (
        torch.stack(
            [
                utils_transform.sixd2matrot(sparse[:, 0:6]),
                utils_transform.sixd2matrot(sparse[:, 6:12]),
                utils_transform.sixd2matrot(sparse[:, 12:18]),
            ],
            dim=1,
        )
        .cpu()
        .numpy()
    )

    # ===== COLORS DEPEND ON GAPS =====
    colors_GT = np.tile(GREEN, (points_GT.shape[0], points_GT.shape[1], 1))
    if gaps is not None:
        # gaps is a TrackingSignalGapsInfo data object
        gaps_list = gaps.gaps
        entities_smpl_idces = gaps.entities_smpl_idces
        for g_list, smpl_idces in zip(gaps_list, entities_smpl_idces):
            for gap in g_list:
                for smpl_idx in smpl_idces:
                    if smpl_idx in SMPL_TO_POINT:
                        colors_GT[gap[0] : gap[1], SMPL_TO_POINT[smpl_idx]] = RED

    # ===== Predicted points extracted from prediction =====
    if pr_body is not None:
        points_pred = pr_body.Jtr[:, to_show[1:]].cpu().numpy()
        colors_pred = np.tile(GRAY_TR, (points_pred.shape[0], points_pred.shape[1], 1))
        rots_pred = np.zeros(
            (points_pred.shape[0], points_pred.shape[1], 3, 3)
        )  # TODO T202407740
        all_points = np.concatenate((points_GT, points_pred), axis=1)
        all_colors = np.concatenate((colors_GT, colors_pred), axis=1)
        all_rots = np.concatenate((rots_GT, rots_pred), axis=1)
    else:
        all_points = points_GT
        all_colors = colors_GT
        all_rots = rots_GT

    return all_points, all_colors, all_rots


class VisualizerWrapper:
    def __init__(self, args, generator, dataset, body_model: BodyModelsWrapper, device):
        self.args = args
        self.generator = generator
        self.dataset = dataset
        self.dataset_name = args.dataset
        self.body_model = body_model
        self.device = device

        self.random = getattr(args, "random", False)
        self.fps = get_dataset_fps(self.dataset_name)

    def _visualize(
        self,
        body_pose,
        body_model,
        fps,
        filename,
        output_dir,
        is_gt=False,
        marker_points: Optional[np.ndarray] = None,
        marker_colors: Optional[np.ndarray] = None,
        marker_rot: Optional[np.ndarray] = None,
        export_results: bool = False,
    ):
        save_filename = filename.split(".")[0].replace("/", "-")
        save_filename = save_filename if not is_gt else save_filename + "_gt"

        # lazy import because we need other dependencies first (bootstrap)
        from utils import utils_visualize

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_video_path = os.path.join(tmpdirname, save_filename + ".mp4")
            stored_paths = utils_visualize.save_animation(
                body_pose=body_pose,
                savepath=temp_video_path,
                bm=body_model.body_model,
                fps=fps,
                resolution=(800, 800),
                marker_points=marker_points,
                marker_colors=marker_colors,
                marker_rot=marker_rot,
                show_rot_axes=True,
                show_origin_axes=False,
                export_meshes=export_results,
            )
            for origin_path in stored_paths:
                dst_path = os.path.join(output_dir, os.path.basename(origin_path))
                pathmgr.copy(src_path=origin_path, dst_path=dst_path, overwrite=True)

    def visualize_animation(
        self, all_info: dict, output_dir: str, filename: str, vis_anim: RollingVisType
    ):
        if vis_anim == RollingVisType.NONE:
            return
        elif vis_anim == RollingVisType.ROLLING:
            plot_rolling_features(all_info, output_dir, filename)
        elif vis_anim == RollingVisType.ROLLING_CR:
            plot_rolling_features_change_rate(all_info, output_dir, filename)
        elif vis_anim == RollingVisType.BOXES_CR:
            plot_change_rate_distribution(all_info, output_dir, filename)
        else:
            raise NotImplementedError

    def visualize_single(
        self,
        sample_idx: int,
        output_dir: str,
        overwrite: bool = False,
        vis_anim: RollingVisType = RollingVisType.NONE,
        num_rep: int = 1,
        export_results: bool = False,
    ):
        gt_dict, cond_dict = self.dataset[sample_idx]
        filename = gt_dict[DataTypeGT.FILENAME]
        gt_data = gt_dict[DataTypeGT.RELATIVE_ROTS]
        sparse = cond_dict[DataTypeGT.SPARSE]
        head_motion = gt_dict[DataTypeGT.HEAD_MOTION]
        gaps = gt_dict[DataTypeGT.TRACKING_GAP]
        gt_gender = gt_dict[DataTypeGT.SMPL_GENDER]
        smpl_model = gt_dict[DataTypeGT.SMPL_MODEL_TYPE]

        filename = f"{filename}_rep{num_rep:02d}"
        store_path = os.path.join(output_dir, filename + ".mp4")
        if not overwrite and pathmgr.exists(store_path):
            logger.info(f"Visualization already exists for {filename}. Skipping...")
            return

        run_anim = vis_anim != RollingVisType.NONE

        output, all_info = self.generator(
            gt_data.unsqueeze(0),
            sparse.unsqueeze(0),
            return_intermediates=run_anim,
            return_predictions=export_results,
            body_model=self.body_model.get_body_model(
                SMPLModelType.SMPLX, SMPLGenderParam.NEUTRAL
            ),
            betas=gt_dict[DataTypeGT.SHAPE_PARAMS][0].unsqueeze(0),
            filenames=[
                gt_dict[DataTypeGT.FILENAME],
            ],
        )

        if run_anim:
            self.visualize_animation(all_info, output_dir, filename, vis_anim)

        local_rot = output[ModelOutputType.RELATIVE_ROTS][0]  # remove batch dimension
        betas = None
        gt_contains_shape_params = DataTypeGT.SHAPE_PARAMS in gt_dict
        model_predicts_shape_params = (
            ModelOutputType.SHAPE_PARAMS in output
            and output[ModelOutputType.SHAPE_PARAMS] is not None
        )
        if gt_contains_shape_params and model_predicts_shape_params:
            # MODE 1: model predicts SHAPE params --› gender is always NEUTRAL for prediction, and betas are the ones predicted by the model
            betas = output[ModelOutputType.SHAPE_PARAMS][0]
            pred_gender = SMPLGenderParam.NEUTRAL
        elif gt_contains_shape_params:
            # MODE 2: used when shape is not predicted, but GT contains shape params --› use GT shape params as these are ASSUMED to be available at runtime
            betas = gt_dict[DataTypeGT.SHAPE_PARAMS]
            pred_gender = gt_gender
        else:
            # MODE 3: retrocompatibility with AGRoL/AvatarPoser benchmark where the shape params are not in GT --› default to 0's, and the gender is always assumed to be MALE
            pred_gender = SMPLGenderParam.MALE

        pr_body = get_body_poses(
            local_rot,
            self.body_model,
            head_motion,
            device=self.device,
            gender=pred_gender,
            model_type=smpl_model,
            betas=betas,
        )
        all_points, all_colors, all_rots = get_marker_points_and_colors(
            gaps, sparse, pr_body
        )

        self._visualize(
            pr_body,
            self.body_model.get_body_model(smpl_model, gt_gender),
            self.fps,
            filename,
            output_dir,
            marker_points=all_points,
            marker_colors=all_colors,
            marker_rot=all_rots,
            export_results=export_results,
        )
        logger.info(f"Visualization generated: {store_path}")

        if export_results:
            predictions = (
                all_info["raw_predictions"] if "raw_predictions" in all_info else None
            )
            if predictions is not None:
                self.export_predictions_at_all_timesteps(
                    filename,
                    output_dir,
                    predictions,
                    head_motion,
                    gt_gender,
                    smpl_model,
                    betas,
                )

            kin_tree = self.body_model.get_kin_tree(smpl_model, gt_gender)[0][:22]
            self.export_skeleton_at_all_timesteps(
                filename,
                output_dir,
                pr_body.Jtr,
                kin_tree,
            )

            self.export_tracking_and_settings_at_all_timesteps(
                filename,
                output_dir,
                sparse,
                predictions,
            )

    def export_predictions_at_all_timesteps(
        self,
        filename: str,
        output_dir: str,
        predictions: th.Tensor,
        head_motion: th.Tensor,
        gender: SMPLGenderParam,
        smpl_model: SMPLModelType,
        betas: th.Tensor,
    ):
        """
        Exports the predictions + tracking signal at all timesteps to a file. Also exports a settings file with info about the sequence.
        - filename: str
        - output_dir: str
        - predictions: th.tensor of shape (seq_len, pred_length, pose_params)
        - head_motion: th.tensor of shape (seq_len, 4, 4) including the orientation and the translation of the head
        - gender: SMPLGenderParam
        - smpl_model: SMPLModelType
        - betas: th.tensor of shape (seq_len, num_betas) containing the SMPL shape parameters
        """
        # ==================== export predicted joints positions at each timestep ====================
        json_dict = {}
        seq_len, pred_length, num_feats = predictions.shape
        predictions = predictions.to(self.device)
        # build betas with pred_length num of frames. We assume betas are constant over the sequence
        betas = betas[0].unsqueeze(0).repeat(pred_length, 1)
        for i in range(seq_len):
            # we use head motion GT to compute the body pose in the future
            # if we approach the end of the sequence, we use the last head motion
            _head_motion = head_motion[i : min(i + pred_length, seq_len)]
            if _head_motion.shape[0] < pred_length:
                diff = pred_length - _head_motion.shape[0]
                # padding with last head motion if we approach the end of the sequence
                _head_motion = th.cat(
                    (_head_motion, _head_motion[-1].unsqueeze(0).repeat(diff, 1, 1)),
                    dim=0,
                )
            pr_body = (
                get_body_poses(
                    predictions[i],
                    self.body_model,
                    _head_motion,
                    device=self.device,
                    gender=gender,
                    model_type=smpl_model,
                    betas=betas,
                )
                .Jtr[:, : constants.NUM_JOINTS_SMPL]
                .cpu()
                .float()
                .numpy()
            )
            frame_dict = {}
            for j in range(pred_length):
                frame_dict[f"{j:02d}"] = pr_body[j, :].reshape(-1).tolist()
            json_dict[f"{i:06d}"] = frame_dict
        # store json
        save_filename = filename.split(".")[0].replace("/", "-")
        # save to temporary file
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "predictions.json")
            with open(tmp_path, "w") as f:
                json.dump(json_dict, f, indent=4, cls=CustomJSONEncoder)
            # copy to manifold
            tgt_path = os.path.join(output_dir, save_filename, "predictions.json")
            pathmgr.copy(
                src_path=tmp_path,
                dst_path=tgt_path,
                overwrite=True,
            )

    def export_skeleton_at_all_timesteps(
        self,
        filename: str,
        output_dir: str,
        pr_skeleton: th.Tensor,
        kin_tree: th.Tensor,
    ):
        """
        Exports the predictions + tracking signal at all timesteps to a file. Also exports a settings file with info about the sequence.
        - filename: str
        - output_dir: str
        - pr_body_skeleton: th.tensor of shape (seq_len, 22, 3) containing the predicted skeleton
        """
        # ==================== export predicted joints positions at each timestep ====================
        json_dict = {"skeleton_parents": kin_tree.tolist(), "skeleton_per_frame": {}}
        seq_len = pr_skeleton.shape[0]

        pr_skeleton = pr_skeleton[:, : constants.NUM_JOINTS_SMPL].cpu().float().numpy()
        all_skeletons = {}
        for i in range(seq_len):
            all_skeletons[f"{i:06d}"] = pr_skeleton[i, :].reshape(-1).tolist()
        json_dict["skeleton_per_frame"] = all_skeletons

        # store json
        save_filename = filename.split(".")[0].replace("/", "-")
        # save to temporary file
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "skeleton.json")
            with open(tmp_path, "w") as f:
                json.dump(json_dict, f, indent=4, cls=CustomJSONEncoder)
            # copy to manifold
            tgt_path = os.path.join(output_dir, save_filename, "skeleton.json")
            pathmgr.copy(
                src_path=tmp_path,
                dst_path=tgt_path,
                overwrite=True,
            )

    def export_tracking_and_settings_at_all_timesteps(
        self,
        filename: str,
        output_dir: str,
        sparse: th.Tensor,
        predictions: Optional[th.Tensor] = None,
    ):
        """
        Exports the tracking signal at all timesteps to a file. Also exports a settings file with info about the sequence.
        - filename: str
        - output_dir: str
        - sparse: th.tensor of shape (seq_len, 54) containing the sparse tracking signal
        - predictions: th.tensor of shape (seq_len, pred_length, pose_params)
        """
        seq_len = sparse.shape[0]
        save_filename = filename.split(".")[0].replace("/", "-")
        # ==================== export tracking signal ====================
        tracking_dict = {f"{i:06d}": {} for i in range(seq_len)}
        for j, (idces_pos, idces_rot) in enumerate(
            zip(
                constants.TRACKING_IDCES_TO_EXPORT_POS[self.dataset_name],
                constants.TRACKING_IDCES_TO_EXPORT_ROT[self.dataset_name],
            )
        ):
            pos = sparse[:, idces_pos]  # (seq_len, 3)
            rot = utils_transform.sixd2quat(sparse[:, idces_rot])  # (seq_len, 4)
            for i in range(seq_len):
                tracking_dict[f"{i:06d}"][f"tracking_{j:02d}"] = (
                    pos[i].cpu().numpy().tolist() + rot[i].cpu().numpy().tolist()
                )  # (x, y, z) + (qw, qx, qy, qz)
        # store json
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "tracking.json")
            with open(tmp_path, "w") as f:
                json.dump(tracking_dict, f, indent=4, cls=CustomJSONEncoder)
            # copy to manifold
            tgt_path = os.path.join(output_dir, save_filename, "tracking.json")
            pathmgr.copy(
                src_path=tmp_path,
                dst_path=tgt_path,
                overwrite=True,
            )

        # ==================== export settings ====================
        settings = {
            "fps": self.fps,
            "folder": output_dir,
            "prediction_length": predictions.shape[1] if predictions is not None else 0,
            "sequence_length": seq_len,
            "num_features": 132,
        }
        # store json as settings
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "settings.json")
            with open(tmp_path, "w") as f:
                json.dump(settings, f, indent=4, cls=CustomJSONEncoder)
            # copy to manifold
            tgt_path = os.path.join(output_dir, save_filename, "settings.json")
            pathmgr.copy(
                src_path=tmp_path,
                dst_path=tgt_path,
                overwrite=True,
            )

    def visualize_single_GT(
        self,
        sample_idx: int,
        output_dir: str,
        overwrite: bool = False,
        export_results: bool = False,
    ):
        gt_dict, cond_dict = self.dataset[sample_idx]
        filename = gt_dict[DataTypeGT.FILENAME]
        gt_data = gt_dict[DataTypeGT.RELATIVE_ROTS]
        sparse = cond_dict[DataTypeGT.SPARSE]
        body_param = gt_dict[DataTypeGT.BODY_PARAMS]
        smpl_gender = gt_dict[DataTypeGT.SMPL_GENDER]
        smpl_model = gt_dict[DataTypeGT.SMPL_MODEL_TYPE]
        store_path = os.path.join(output_dir, filename + "_gt.mp4")
        if not overwrite and pathmgr.exists(store_path):
            logger.info(f"Visualization already exists for {filename}. Skipping...")
            return

        gt_body = get_GT_body_poses(
            self.body_model,
            body_param,
            gt_data.shape[0],
            self.device,
            smpl_gender,
            smpl_model,
        )
        all_points, all_colors, all_rots = get_marker_points_and_colors(
            None, sparse, gt_body
        )
        self._visualize(
            gt_body,
            self.body_model.get_body_model(smpl_model, smpl_gender),
            self.fps,
            filename,
            output_dir,
            is_gt=True,
            marker_points=all_points,
            marker_colors=all_colors,
            marker_rot=all_rots,
            export_results=export_results,
        )
        logger.info(f"GT visualization generated: {store_path}")
        if export_results:
            kin_tree = self.body_model.get_kin_tree(smpl_model, smpl_gender)[0][:22]
            self.export_skeleton_at_all_timesteps(
                filename + "_gt",
                output_dir,
                gt_body.Jtr,
                kin_tree,
            )

            self.export_tracking_and_settings_at_all_timesteps(
                filename + "_gt",
                output_dir,
                sparse,
            )

    def visualize_all(
        self,
        output_dir: str,
        samples: Optional[List[str]] = None,
        gt_data: bool = False,
        overwrite: bool = False,
        vis_anim: RollingVisType = RollingVisType.NONE,
        num_rep: int = 1,
        export_results: bool = False,
    ):
        if samples is None:
            dataset_len = len(self.dataset)
            if self.random:
                order = np.random.permutation(dataset_len)
            else:
                order = np.arange(dataset_len)
        else:
            logger.info(f"Visualizing {len(samples)} samples: {samples}")
            order = samples

        with torch.no_grad():
            for sample_index in tqdm(order):
                if gt_data:
                    self.visualize_single_GT(
                        sample_index,
                        output_dir,
                        overwrite=overwrite,
                        export_results=export_results,
                    )
                else:
                    self.visualize_single(
                        sample_index,
                        output_dir,
                        overwrite=overwrite,
                        vis_anim=vis_anim,
                        num_rep=num_rep,
                        export_results=export_results,
                    )

    def visualize_by_names(
        self,
        output_dir: str,
        names: List[str],
        gt_data: bool = False,
        overwrite: bool = False,
        vis_anim: RollingVisType = RollingVisType.NONE,
        num_rep: int = 1,
        export_results: bool = False,
    ):
        tgt_names = set(names)
        samples_list = []
        for i, c_filename in enumerate(self.dataset.filename_list):
            if c_filename in tgt_names:
                samples_list.append(i)

        self.visualize_all(
            output_dir,
            samples=samples_list,
            gt_data=gt_data,
            overwrite=overwrite,
            vis_anim=vis_anim,
            num_rep=num_rep,
            export_results=export_results,
        )

    def visualize_subset(
        self,
        output_dir: str,
        gt_data: bool = False,
        overwrite: bool = False,
        vis_anim: RollingVisType = RollingVisType.NONE,
        num_rep: int = 1,
        export_results: bool = False,
    ):
        assert self.dataset_name in VIS_SAMPLES_SUBSET, "No subset defined for dataset"
        self.visualize_by_names(
            output_dir,
            names=VIS_SAMPLES_SUBSET[self.dataset_name],
            gt_data=gt_data,
            overwrite=overwrite,
            vis_anim=vis_anim,
            num_rep=num_rep,
            export_results=export_results,
        )


def get_min_from_intermediates(all_info, sample_idx=0):
    all_y_min, all_y_max = [], []
    gt = all_info["gt"][sample_idx]
    gt_y_min, gt_y_max = gt.min(axis=0), gt.max(axis=0)
    for feat_idx in range(gt_y_min.shape[0]):
        # find max in intermediates
        for intermediate in all_info["intermediates"]:
            y_min = min(
                gt_y_min[feat_idx],
                intermediate[2][sample_idx, :, feat_idx].min(),
                intermediate[3][sample_idx, :, feat_idx].min(),
            )
            y_max = max(
                gt_y_max[feat_idx],
                intermediate[2][sample_idx, :, feat_idx].max(),
                intermediate[3][sample_idx, :, feat_idx].max(),
            )
        all_y_min.append(y_min)
        all_y_max.append(y_max)
    return all_y_min, all_y_max


def plot_rolling_features(
    all_info, output_dir, filename, sample_idx=0, feats=(0, 4, 5, 20, 21)
):
    save_filename = filename.split(".")[0].replace("/", "-")
    save_path = os.path.join(
        output_dir,
        save_filename + f"_rolling_vis_{'_'.join([str(f) for f in feats])}.mp4",
    )
    if pathmgr.exists(save_path):
        logger.info("{} already exists, skipping".format(save_path))
        return

    # 0 --> global orientation
    # 1, 2 --> left, right hip
    # 4, 5 --> left, right knee
    # 7, 8 --> left, right ankle
    # 10, 11 --> left, right foot
    # 18, 19 --> left, right elbow
    # 20, 21 --> left, right wrist
    gt = all_info["gt"][sample_idx]
    all_y_min, all_y_max = get_min_from_intermediates(all_info, sample_idx=sample_idx)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(1, len(feats), figsize=(5 * len(feats), 5))

    def animate(i):
        # plot as many plots as there are features
        for cax, num_feat in zip(ax, feats):
            cax.clear()

            idx = all_info["intermediates"][i][0]
            pred = all_info["intermediates"][i][2][sample_idx, :, num_feat]
            noisy = all_info["intermediates"][i][3][sample_idx, :, num_feat]
            xs_pred = np.arange(idx, min(gt.shape[0], idx + pred.shape[0]))

            # Highlight the current segment of the GT
            cax.plot(xs_pred, noisy[: len(xs_pred)], label="Noisy pred", color="gray")
            cax.plot(gt[:, num_feat], label="Ground Truth", color="black")
            cax.plot(
                xs_pred, pred[: len(xs_pred)], color="red", lw=2, label="Prediction"
            )
            if all_info["intermediates"][i][1].shape[1] > 0:
                ctx = all_info["intermediates"][i][1][sample_idx, :, num_feat]
                xs_ctx = np.arange(idx - ctx.shape[0], idx)
                cax.plot(xs_ctx, ctx, color="blue", lw=2, label="Context")

            cax.set_title(f"Frame {idx}/{gt.shape[0]}")
            cax.set_xlim(0, gt.shape[0])
            cax.set_ylim(all_y_min[num_feat], all_y_max[num_feat])
            cax.set_xlabel("Frame")
            cax.set_ylabel(f"Motion feat: {num_feat}")
            cax.legend(loc="upper right")
        plt.draw()

    # Create the animation
    anim = FuncAnimation(fig, animate, len(all_info["intermediates"]))

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_video_path = os.path.join(tmpdirname, save_filename + "_rolling_vis.mp4")
        # Store to video
        anim.save(temp_video_path, writer="ffmpeg", fps=10)
        pathmgr.copy(
            src_path=temp_video_path,
            dst_path=save_path,
        )


def plot_rolling_features_change_rate(
    all_info, output_dir, filename, sample_idx=0, feats=(0, 4, 5, 20, 21)
):
    save_filename = filename.split(".")[0].replace("/", "-")
    save_path = os.path.join(
        output_dir,
        save_filename + f"_rolling_vis_CR_{'_'.join([str(f) for f in feats])}.mp4",
    )
    if pathmgr.exists(save_path):
        logger.info("{} already exists, skipping".format(save_path))
        return

    gt = all_info["gt"][sample_idx]
    prediction = all_info["prediction"][sample_idx]
    all_y_min, all_y_max = get_min_from_intermediates(all_info, sample_idx=sample_idx)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(1, len(feats), figsize=(5 * len(feats), 5))
    ax2 = []
    for cax in ax:
        ax2.append(cax.twinx())

    def animate(i):
        # plot as many plots as there are features
        for cax, cax2, num_feat in zip(ax, ax2, feats):
            cax.clear()
            cax2.clear()

            idx = all_info["intermediates"][i][0]
            pred = all_info["intermediates"][i][2][sample_idx, :, num_feat]
            xs_pred = list(np.arange(idx, min(gt.shape[0], idx + pred.shape[0])))

            if i > 1 and i < len(all_info["intermediates"]) - 1:
                prev_pred = all_info["intermediates"][i - 1][2][sample_idx, :, num_feat]
                M = len(xs_pred)
                correction = abs(pred[: M - 1] - prev_pred[1:M])
                cax2.bar(
                    xs_pred[:-1],
                    correction,
                    label="Correction",
                    color="gray",
                    alpha=0.7,
                )
                cax2.set_ylim(0, 0.1)

            # Highlight the current segment of the GT
            cax.plot(gt[:, num_feat], label="Ground Truth", color="black")
            cax.plot(
                xs_pred, pred[: len(xs_pred)], color="red", lw=2, label="Prediction"
            )
            if all_info["intermediates"][i][1].shape[1] > 0:
                # plot all past prediction up until this frame
                cax.plot(
                    np.arange(0, idx),
                    prediction[:idx, num_feat],
                    color="orange",
                    lw=2,
                    label="Past motion",
                )
                # plot context fed to the model
                ctx = all_info["intermediates"][i][1][sample_idx, :, num_feat]
                xs_ctx = np.arange(idx - ctx.shape[0], idx)
                cax.plot(xs_ctx, ctx, color="blue", lw=2, label="Context")

            cax.set_title(f"Frame {idx}/{gt.shape[0]}")
            cax.set_xlim(0, gt.shape[0])
            cax.set_ylim(all_y_min[num_feat], all_y_max[num_feat])
            cax.set_xlabel("Frame")
            cax.set_ylabel(f"Motion feat: {num_feat}")
            cax.legend(loc="upper right")
        plt.draw()

    # Create the animation
    anim = FuncAnimation(fig, animate, len(all_info["intermediates"]))

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_video_path = os.path.join(tmpdirname, save_filename + "_rolling_vis.mp4")
        # Store to video
        anim.save(temp_video_path, writer="ffmpeg", fps=10)
        pathmgr.copy(
            src_path=temp_video_path,
            dst_path=save_path,
        )


def plot_change_rate_distribution(
    all_info, output_dir, filename, sample_idx=0, feats=(0, 4, 5, 20, 21)
):
    save_filename = filename.split(".")[0].replace("/", "-")
    save_path = os.path.join(
        output_dir,
        save_filename + "_change_rate_dist.png",
    )

    gt = all_info["gt"][sample_idx]
    fig, ax = plt.subplots(1, len(feats), figsize=(5 * len(feats), 5))

    for cax, num_feat in zip(ax, feats):
        data = []
        for i in range(len(all_info["intermediates"])):
            idx = all_info["intermediates"][i][0]
            pred = all_info["intermediates"][i][2][sample_idx, :, num_feat]
            xs_pred = list(np.arange(idx, min(gt.shape[0], idx + pred.shape[0])))

            if i > 1 and i < len(all_info["intermediates"]) - 1:
                prev_pred = all_info["intermediates"][i - 1][2][sample_idx, :, num_feat]
                M = len(xs_pred)
                correction = abs(pred[: M - 1] - prev_pred[1:M])
                for j, c in enumerate(correction):
                    if len(data) <= j:
                        data.append([])
                    data[j].append(c)

        cax.boxplot(data, showfliers=False)
        cax.set_title(f"Motion feat: {num_feat}")
        cax.set_xlabel("Future frames")
        cax.set_ylabel("Angle")
        cax.legend(loc="upper right")

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_path = os.path.join(tmpdirname, save_filename + "_img.png")
        plt.savefig(temp_path)
        pathmgr.copy(
            src_path=temp_path,
            dst_path=save_path,
        )
