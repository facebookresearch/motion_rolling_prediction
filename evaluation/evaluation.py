# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import torch
from evaluation.utils import (
    BodyModelsWrapper,
    get_body_poses,
    get_dataset_fps,
    get_GT_body_poses,
    get_min_frames_to_eval,
)

from loguru import logger

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from tqdm import tqdm

from utils.constants import DataTypeGT, ModelOutputType, SMPLGenderParam, SMPLModelType
from utils.metrics import (
    AccumulateArray,
    AverageValue,
    from_gaps_to_masks,
    get_all_metrics,
    get_all_metrics_trackingloss,
    get_metric_function,
    get_plots_configs,
    is_array_based_metric,
    keep_logging_metrics,
    MetricsInputData,
    remove_frames_to_gaps,
)

from pathlib import Path

def padding_collate(batch):
    """
    Receives a list with #BATCH_SIZE tuples (gt_data, cond_data)
    It pads the sequences to the same length (only data structures specified in keys_to_pad)
    - All 'key_to_pad' are returned as tensors of shape (BATCH_SIZE, MAX_SEQ_LEN, ...)
    - All others are returned as lists of length BATCH_SIZE
    """
    keys_to_pad = {
        DataTypeGT.RELATIVE_ROTS,
        DataTypeGT.SPARSE,
        DataTypeGT.HEAD_MOTION,
    }
    assert len(batch) > 0, "Batch is empty"
    gt_data, cond_data = {}, {}
    for i, data_dict in enumerate((gt_data, cond_data)):
        # pad the sequences to the same length
        for k in batch[0][i].keys():
            if k in keys_to_pad:
                data_dict[k] = pad_sequence([x[i][k] for x in batch], batch_first=True)
            else:
                data_dict[k] = [x[i][k] for x in batch]
    return gt_data, cond_data


class SortedSampler(Sampler):
    """Samples elements sequentially sorted by the sequence length, and filtered by min seq length."""

    def __init__(self, data_source, min_frames):
        self.data_source = data_source
        self.min_frames = min_frames

        # Filter and sort in one step
        self.filtered_and_sorted_indices = sorted(
            (
                idx
                for idx in range(len(self.data_source))
                if self.data_source[idx][0][DataTypeGT.NUM_FRAMES] >= self.min_frames
            ),
            key=lambda x: self.data_source[x][0][DataTypeGT.NUM_FRAMES],
        )

        num_filtered = len(self.data_source) - len(self.filtered_and_sorted_indices)
        logger.info(f"SortedSampler: {num_filtered} filtered, {len(self)} remaining")

    def __iter__(self):
        return iter(self.filtered_and_sorted_indices)

    def __len__(self):
        return len(self.filtered_and_sorted_indices)


class EvaluatorWrapper:
    def __init__(
        self,
        args,
        generator,
        dataset,
        body_model: BodyModelsWrapper,
        device,
        batch_size=1,
    ):
        self.args = args
        self.generator = generator
        self.dataset = dataset
        self.dataset_name = args.dataset
        self.device = device
        self.body_model = body_model
        self.fps = get_dataset_fps(self.dataset_name)
        self.MIN_FRAMES_TO_EVAL = get_min_frames_to_eval(self.dataset_name)
        self.batch_size = batch_size

        # initialize data loader. Sorting the dataset by sequence length to
        # speed up the evaluation, as sequences with similar length will be
        # processed together, thus minimizing the padding (i.e., useless computation).
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=padding_collate,
            sampler=SortedSampler(self.dataset, min_frames=self.MIN_FRAMES_TO_EVAL + 3),
        )

    def evaluate_from_prediction(
        self,
        output,
        body_param_GT,
        head_motion,
        gaps,
        pred_gender: SMPLGenderParam,
        gt_gender: SMPLGenderParam,
        model_type: SMPLModelType,
        betas=None,
    ):
        pr_body = get_body_poses(
            output,
            self.body_model,
            head_motion,
            device=self.device,
            gender=pred_gender,
            model_type=model_type,
            betas=betas,
        )

        gt_body = get_GT_body_poses(
            self.body_model,
            body_param_GT,
            output.shape[0],
            self.device,
            gt_gender,
            model_type,
        )

        pr_pos = pr_body.Jtr[self.MIN_FRAMES_TO_EVAL - 1 :, :22, :]
        pr_angle = pr_body.full_pose.reshape(pr_body.Jtr.shape)[
            self.MIN_FRAMES_TO_EVAL - 1 :, :22
        ]

        gt_pos = gt_body.Jtr[self.MIN_FRAMES_TO_EVAL - 1 :, :22, :]
        gt_angle = gt_body.full_pose.reshape(pr_body.Jtr.shape)[
            self.MIN_FRAMES_TO_EVAL - 1 :, :22
        ]

        metrics_sets = [
            get_all_metrics(),
        ]
        masks, processed_gaps = None, None
        if gaps is not None:
            metrics_sets.append(get_all_metrics_trackingloss())
            masks = from_gaps_to_masks(
                gaps, output.shape[0], self.MIN_FRAMES_TO_EVAL - 1
            )
            processed_gaps = remove_frames_to_gaps(gaps, self.MIN_FRAMES_TO_EVAL - 1)

        metrics_input_data = MetricsInputData(
            pred_positions=pr_pos,
            pred_angles=pr_angle,
            pred_mesh=pr_body.v[self.MIN_FRAMES_TO_EVAL - 1 :],
            gt_positions=gt_pos,
            gt_angles=gt_angle,
            gt_mesh=gt_body.v[self.MIN_FRAMES_TO_EVAL - 1 :],
            fps=self.fps,
            trackingloss_masks=masks,
            gaps=processed_gaps,
        )
        eval_log = {}
        for metrics in metrics_sets:
            for metric in metrics:
                eval_log[metric] = (
                    get_metric_function(metric)(metrics_input_data).cpu().numpy()
                )
        return eval_log

    def evaluate_single(self, sample_idx):
        gt_dict, cond_dict = self.dataset[sample_idx]
        gt_data = gt_dict[DataTypeGT.RELATIVE_ROTS]
        sparse = cond_dict[DataTypeGT.SPARSE]
        body_param_GT = gt_dict[DataTypeGT.BODY_PARAMS]
        head_motion = gt_dict[DataTypeGT.HEAD_MOTION]
        gaps = gt_dict[DataTypeGT.TRACKING_GAP]
        gt_gender = gt_dict[DataTypeGT.SMPL_GENDER]
        model_type = gt_dict[DataTypeGT.SMPL_MODEL_TYPE]

        output, _ = self.generator(
            gt_data.unsqueeze(0),
            sparse.unsqueeze(0),
            return_intermediates=False,
            body_model=self.body_model.get_body_model(
                SMPLModelType.SMPLX, SMPLGenderParam.NEUTRAL
            ),
            betas=torch.stack([b[0] for b in gt_dict[DataTypeGT.SHAPE_PARAMS]], 0),
            filenames=gt_dict[DataTypeGT.FILENAME],
        )
        local_rot = output[ModelOutputType.RELATIVE_ROTS][0]
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

        return self.evaluate_from_prediction(
            local_rot,
            body_param_GT,
            head_motion,
            gaps,
            pred_gender,
            gt_gender,
            model_type,
            betas=betas,
        )

    def evaluate_all(self):
        log = {}
        fine_grained_results = []
        total_samples = 0
        with torch.no_grad():
            for gt_dict, cond_dict in tqdm(self.data_loader):
                gt_data = gt_dict[DataTypeGT.RELATIVE_ROTS]
                sparse = cond_dict[DataTypeGT.SPARSE]
                body_param = gt_dict[DataTypeGT.BODY_PARAMS]
                head_motion = gt_dict[DataTypeGT.HEAD_MOTION]
                gaps = gt_dict[DataTypeGT.TRACKING_GAP]
                num_frames = gt_dict[DataTypeGT.NUM_FRAMES]
                filenames = gt_dict[DataTypeGT.FILENAME]
                gt_genders = gt_dict[DataTypeGT.SMPL_GENDER]
                model_types = gt_dict[DataTypeGT.SMPL_MODEL_TYPE]

                batch_size = len(num_frames)

                # inference
                output, _ = self.generator(
                    gt_data,
                    sparse,
                    return_intermediates=False,
                    body_model=self.body_model.get_body_model(
                        SMPLModelType.SMPLX, SMPLGenderParam.NEUTRAL
                    ),
                    betas=torch.stack(
                        [b[0] for b in gt_dict[DataTypeGT.SHAPE_PARAMS]], 0
                    ),
                    filenames=filenames,
                )

                local_rot = output[ModelOutputType.RELATIVE_ROTS]
                betas = None
                if (
                    DataTypeGT.SHAPE_PARAMS in gt_dict
                    and ModelOutputType.SHAPE_PARAMS in output
                    and output[ModelOutputType.SHAPE_PARAMS] is not None
                ):
                    # new version of the dataset that has used shape params --> use NEUTRAL
                    betas = output[ModelOutputType.SHAPE_PARAMS]
                    pred_genders = [
                        SMPLGenderParam.NEUTRAL,
                    ] * batch_size
                elif DataTypeGT.SHAPE_PARAMS in gt_dict:
                    # new version + no params predicted -- use GT
                    betas = gt_dict[DataTypeGT.SHAPE_PARAMS]
                    pred_genders = gt_genders
                else:
                    # old version of dataset where MALE is default, and shape is not predictable
                    pred_genders = [
                        SMPLGenderParam.MALE,
                    ] * batch_size

                # sequentially evaluate (FK not batcherized)
                for i in range(batch_size):
                    if betas is not None:
                        betas_ = betas[i][: num_frames[i]]
                    else:
                        betas_ = None

                    # we slice the output to the number of frames in the sequence, to remove the padding introduced in the padding_collate function
                    instance_log = self.evaluate_from_prediction(
                        local_rot[i][: num_frames[i]],
                        body_param[i],
                        head_motion[i][: num_frames[i]],
                        gaps[i],
                        pred_genders[i],
                        gt_genders[i],
                        model_types[i],
                        betas=betas_,
                    )

                    for key in instance_log:
                        if key not in log:
                            log[key] = (
                                AverageValue()
                                if not is_array_based_metric(key)
                                else AccumulateArray()
                            )
                        log[key].add_value(instance_log[key])

                    metrics_list_to_csv = [
                        instance_log[k]
                        for k in log.keys()
                        if not is_array_based_metric(k)
                    ]
                    fine_grained_results.append(
                        [filenames[i], num_frames[i]] + metrics_list_to_csv
                    )
                    total_samples += 1

        df_titles = ["filename", "num_frames"] + [
            k for k in log.keys() if not is_array_based_metric(k)
        ]
        fine_grained_df = pd.DataFrame(fine_grained_results, columns=df_titles)

        arr_based_metrics = {}
        summary_log = {}
        for metric in log.keys():
            if is_array_based_metric(metric):
                # store plot + npy file with all values
                arr_based_metrics[metric] = log[metric].get_array()
            else:
                summary_log[metric] = log[metric].get_average()

        return summary_log, fine_grained_df, arr_based_metrics

    def store_all_results(self, df: pd.DataFrame, csv_path: Path):
        df.to_csv(csv_path, index=False)
        logger.info(f"Results successfully stored in a csv file: {csv_path=}")

    def store_plots(self, metrics: dict, plot_dir: Path):
        if len(metrics.keys()) == 0:
            return

        for plot_cfg in get_plots_configs():
            # plot_cfg is PlotConfig data class from utils.metrics
            title = plot_cfg.title
            all_metrics_needed = plot_cfg.curve_styles.keys()
            if not all(m in metrics for m in all_metrics_needed):
                logger.warning(
                    f"Skipping plot '{title}' because not all metrics are available: {all_metrics_needed}"
                )
                continue
            plt.clf()
            plt.title(title)
            abs_max = -float("inf")
            for metric_name, style in plot_cfg.curve_styles.items():
                if metric_name not in metrics:
                    continue
                y_values = list(
                    metrics[metric_name].mean(axis=0)
                )  # from [samples, T] --> [T]
                plt.plot(y_values, label=style.label)
                abs_max = max(max(y_values), abs_max)
                plt.ylim([0, 1.1 * abs_max])
                # store array to npz
                filename = "arr_" + metric_name + ".npz"
                np.savez(plot_dir / filename, values=metrics[metric_name])
            plt.legend()
            plt.xlabel("Frame")
            plt.ylabel("Value")

            tgt_path = plot_dir / (plot_cfg.filename + ".png")
            plt.savefig(tgt_path)
            plt.close()
            logger.info(f"'{title}' plot saved in {tgt_path}")

    def print_results(self, log):
        # print the value for all the metrics
        logger.info("Metrics:")
        for metric in log.keys():
            logger.info(f"{metric}: {log[metric]}")

    def push_to_tb(self, log, tb_writer, iteration, suffix=""):
        logging_metrics = keep_logging_metrics(log)
        for metric in logging_metrics:
            tb_writer.add_scalar(
                f"eval{suffix}/{metric}",
                log[metric],
                iteration,
            )
