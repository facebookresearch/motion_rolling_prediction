# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Metric functions with same inputs

import math
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial
from typing import Dict, Optional

import numpy as np
import torch as th
from data_loaders.dataloader import TrackingSignalGapsInfo
from utils.constants import TransitionType

FLOOR_HEIGHT = 0
UP_DIM = 2


@dataclass
class TrackingLossMasks:
    tracking_loss: th.Tensor
    rec_frame: th.Tensor
    loss_frame: th.Tensor


@dataclass
class MetricsInputData:
    pred_positions: th.Tensor
    pred_angles: th.Tensor
    pred_mesh: th.Tensor
    gt_positions: th.Tensor
    gt_angles: th.Tensor
    gt_mesh: th.Tensor
    fps: float
    trackingloss_masks: Optional[TrackingLossMasks]
    gaps: Optional[TrackingSignalGapsInfo]


class RestrictType(str, Enum):
    NONE = auto()
    LOSS = auto()
    REC = auto()
    REC_FRAME = auto()
    LOSS_FRAME = auto()


BODY_ENTITIES = {
    "all": list(range(22)),
    "hands": [20, 21],
    "upper": [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "lower": [0, 1, 2, 4, 5, 7, 8, 10, 11],
    "root": [0],
}

METERS_TO_CENTIMETERS = 100.0
RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)  # 57.2958 grads
REC_F_LENGTH = 3
LOSS_F_LENGTH = 3


class AverageValue:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def add_value(self, value):
        if not np.isnan(value):
            self.sum += value
            self.count += 1

    def get_average(self):
        if self.count == 0:
            return np.nan
        return self.sum / self.count


class AccumulateArray:
    def __init__(self):
        self.array = None

    def add_value(self, arr):
        # vector --> (elements, ...)
        if self.array is None:
            self.array = arr
        elif arr.shape[0] > 0:
            self.array = np.concatenate((self.array, arr), axis=0)

    def get_array(self):
        return self.array


def from_gaps_to_masks(
    gaps: TrackingSignalGapsInfo, total_frames: int, frames_to_remove: int
) -> TrackingLossMasks:
    # gaps mask
    tl_mask = th.zeros(total_frames, dtype=th.bool)
    for gap_set in gaps.gaps:
        for gap in gap_set:
            tl_mask[gap[0] : gap[1]] = True
    tl_mask = tl_mask[frames_to_remove:]

    # recovery frame mask
    recf_mask = th.zeros(total_frames, dtype=th.bool)
    for gap_set in gaps.gaps:
        for gap in gap_set:
            recf_mask[gap[1] : min(gap[1] + REC_F_LENGTH, total_frames)] = True
    recf_mask = recf_mask[frames_to_remove:]

    # recovery frame mask
    lossf_mask = th.zeros(total_frames, dtype=th.bool)
    for gap_set in gaps.gaps:
        for gap in gap_set:
            lossf_mask[gap[0] : min(gap[0] + LOSS_F_LENGTH, total_frames)] = True
    lossf_mask = lossf_mask[frames_to_remove:]
    return TrackingLossMasks(tl_mask, recf_mask, lossf_mask)


def remove_frames_to_gaps(
    gaps: TrackingSignalGapsInfo, frames_to_remove: int
) -> TrackingSignalGapsInfo:
    """
    Return a new gaps object considering that the first min_frames_to_eval frames are removed.
    """
    new_gaps = []
    for gap_set in gaps.gaps:
        new_gap_set = []
        for gap in gap_set:
            if gap[1] < frames_to_remove:
                continue
            new_init = max(0, gap[0] - frames_to_remove)
            new_end = gap[1] - frames_to_remove
            new_gap_set.append((new_init, new_end))
        new_gaps.append(new_gap_set)
    return TrackingSignalGapsInfo(
        new_gaps, gaps.entities_idces, gaps.entities_smpl_idces
    )


def restrict_and_mean(
    values: th.Tensor, restrict: RestrictType, masks: Optional[TrackingLossMasks] = None
):
    """
    Function to restrict the values to a subset of frames from the sequence and then average the values.
    The subset is defined by the restrict parameter:
    - RestrictType.NONE: no restriction, all frames are used
    - RestrictType.LOSS: only frames under tracking loss are used (i.e., lost tracking signal)
    - RestrictType.REC: only frames with tracking signal available (i.e., recovered tracking signal)
    - RestrictType.REC_FRAME: only the frame right after recovering the tracking signal
    """
    if restrict == RestrictType.NONE:
        return values.mean()

    assert masks is not None, "masks must be provided if restrict is not None"
    if restrict == RestrictType.LOSS:
        mask = masks.tracking_loss
    elif restrict == RestrictType.REC:
        mask = ~masks.tracking_loss
    elif restrict == RestrictType.REC_FRAME:
        mask = masks.rec_frame
    elif restrict == RestrictType.LOSS_FRAME:
        mask = masks.loss_frame
    else:
        raise ValueError(f"unknown restrict {restrict}")

    diff = mask.shape[0] - values.shape[0]
    mask = mask[diff:]  # for velocity or jitter, where some padding is needed
    return values[mask].mean()


def transition_jerk(
    data: MetricsInputData,
    target: str = "pred",
    joints: str = "all",
    duration: float = 0.5,
    transition_type: TransitionType = TransitionType.S_TO_T,
):
    """
    Computes the jerk inside a margin of #duration seconds to the right of the transitioning
    frame.
    - input positions format --> [T, 22, 3]
    """
    gaps = data.gaps
    assert gaps is not None, "gaps must be provided if computing transition jerk"
    assert target in ["pred", "gt"], "target must be either pred or gt"
    assert joints in BODY_ENTITIES, f"unknown '{joints}' entity"
    length_f = int(duration * data.fps)
    motion = data.pred_positions if target == "pred" else data.gt_positions

    if motion.shape[0] - 3 < length_f:
        return th.zeros((0, length_f), device=motion.device)

    motion = motion[..., BODY_ENTITIES[joints], :]
    jitter = (motion[3:] - 3 * motion[2:-1] + 3 * motion[1:-2] - motion[:-3]) * (
        data.fps**3
    )
    # abs for each joints
    jitter = jitter.norm(p=2, dim=2)
    # average across joints
    jitter = jitter.max(dim=1)[0]
    # jitter is now [T-3, 1]
    all_jitter_segs = []
    for gap_set in gaps.gaps:
        for gap in gap_set:
            # gap is [start, end] of a S period
            gap = (
                gap[0] - 3,
                gap[1] - 3,
            )  # shift to the left to match the jitter shape
            gap_len = gap[1] - gap[0]
            if (
                transition_type == TransitionType.T_TO_S
                and gap[0] > 0
                and gap_len >= length_f
            ):
                # we get left-most #length_f frames inside the gap
                all_jitter_segs.append(jitter[gap[0] : gap[0] + length_f])
            elif (
                transition_type == TransitionType.S_TO_T
                and gap[1] > 0
                and gap[1] + length_f < jitter.shape[0]
            ):
                # we get the first #length_f frames after the gap
                all_jitter_segs.append(jitter[gap[1] : gap[1] + length_f])
    if len(all_jitter_segs) == 0:
        return th.zeros((0, length_f), device=motion.device)
    return th.stack(all_jitter_segs, dim=0).to(motion.device)


def jitter(
    data: MetricsInputData,
    target: str = "pred",
    joints: str = "all",
    restrict: RestrictType = RestrictType.NONE,
):
    assert target in ["pred", "gt"], "target must be either pred or gt"
    assert joints in BODY_ENTITIES, f"unknown '{joints}' entity"
    motion = data.pred_positions if target == "pred" else data.gt_positions
    motion = motion[..., BODY_ENTITIES[joints], :]
    jitter = (
        (motion[3:] - 3 * motion[2:-1] + 3 * motion[1:-2] - motion[:-3]) * (data.fps**3)
    ).norm(dim=2)
    return restrict_and_mean(jitter, restrict, data.trackingloss_masks)


def mpjre(
    data: MetricsInputData,
    joints: str = "all",
    restrict: RestrictType = RestrictType.NONE,
    **kwargs,
):
    assert joints in BODY_ENTITIES, f"unknown '{joints}' entity"
    predicted_angle = data.pred_angles[..., BODY_ENTITIES[joints], :].flatten(
        start_dim=-2
    )
    gt_angle = data.gt_angles[..., BODY_ENTITIES[joints], :].flatten(start_dim=-2)

    diff = gt_angle - predicted_angle
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    rot_error = th.absolute(diff)
    result = restrict_and_mean(rot_error, restrict, data.trackingloss_masks)
    return result * RADIANS_TO_DEGREES


def mpjpe(
    data: MetricsInputData,
    joints: str = "all",
    restrict: RestrictType = RestrictType.NONE,
):
    assert joints in BODY_ENTITIES, f"unknown '{joints}' entity"
    predicted_position = data.pred_positions[..., BODY_ENTITIES[joints], :]
    gt_position = data.gt_positions[..., BODY_ENTITIES[joints], :]
    pos_error = th.sqrt(th.sum(th.square(gt_position - predicted_position), dim=-1))
    result = restrict_and_mean(pos_error, restrict, data.trackingloss_masks)
    return result * METERS_TO_CENTIMETERS


def mpjve(
    data: MetricsInputData,
    joints: str = "all",
    restrict: RestrictType = RestrictType.NONE,
):
    gt_velocity = (data.gt_positions[1:, ...] - data.gt_positions[:-1, ...]) * data.fps
    predicted_velocity = (
        data.pred_positions[1:, ...] - data.pred_positions[:-1, ...]
    ) * data.fps
    vel_error = th.sqrt(th.sum(th.square(gt_velocity - predicted_velocity), dim=-1))

    result = restrict_and_mean(vel_error, restrict, data.trackingloss_masks)
    return result * METERS_TO_CENTIMETERS


METRIC_FUNCS_DICT = {
    "mpjre": mpjre,
    "mpjpe": mpjpe,
    "mpjve": mpjve,
    #"handpe": partial(mpjpe, joints="hands"),
    #"upperpe": partial(mpjpe, joints="upper"),
    #"lowerpe": partial(mpjpe, joints="lower"),
    #"rootpe": partial(mpjpe, joints="root"),
    "pred_jitter": partial(jitter, joints="all", target="pred"),
    #"pred_H_jitter": partial(jitter, joints="hands", target="pred"),
    #"pred_UB_jitter": partial(jitter, joints="upper", target="pred"),
    #"pred_LB_jitter": partial(jitter, joints="lower", target="pred"),
    "gt_jitter": partial(jitter, joints="all", target="gt"),
    #"gt_H_jitter": partial(jitter, joints="hands", target="gt"),
    #"gt_UB_jitter": partial(jitter, joints="upper", target="gt"),
    #"gt_LB_jitter": partial(jitter, joints="lower", target="gt"),
}

ARRAY_BASED_METRICS = ["S_to_T_jerk", "T_to_S_jerk", "gt_S_to_T_jerk", "gt_T_to_S_jerk"]


@dataclass
class CurveStyle:
    label: str
    color: str
    linestyle: str


@dataclass
class PlotConfig:
    # config for the plot, which includes all curve styles
    title: str
    curve_styles: Dict[str, CurveStyle]
    filename: str


PLOT_STYLE_GT = CurveStyle("Predicted motion", "red", "-")
PLOT_STYLE_PRED = CurveStyle("GT motion", "black", "--")

METRICS_PLOTS_CONFIGS = [
    PlotConfig(
        "S->T transition",
        {
            "S_to_T_jerk": PLOT_STYLE_GT,
            "gt_S_to_T_jerk": PLOT_STYLE_PRED,
        },
        "S_to_T_transition",
    ),
    PlotConfig(
        "T->S transition",
        {
            "T_to_S_jerk": PLOT_STYLE_GT,
            "gt_T_to_S_jerk": PLOT_STYLE_PRED,
        },
        "T_to_S_transition",
    ),
]

METRIC_FUNCS_DICT_TRACKING_LOSS = {
    "S_to_T_jerk": partial(
        transition_jerk,
        joints="all",
        target="pred",
        transition_type=TransitionType.S_TO_T,
    ),
    "T_to_S_jerk": partial(
        transition_jerk,
        joints="all",
        target="pred",
        transition_type=TransitionType.T_TO_S,
    ),
    "gt_S_to_T_jerk": partial(
        transition_jerk,
        joints="all",
        target="gt",
        transition_type=TransitionType.S_TO_T,
    ),
    "gt_T_to_S_jerk": partial(
        transition_jerk,
        joints="all",
        target="gt",
        transition_type=TransitionType.T_TO_S,
    ),
    # "tl_mpjpe": partial(mpjpe, restrict=RestrictType.LOSS),
    # "tr_mpjpe": partial(mpjpe, restrict=RestrictType.REC),
    # "tl_mpjve": partial(mpjve, restrict=RestrictType.LOSS),
    # "tr_mpjve": partial(mpjve, restrict=RestrictType.REC),
    # "tl_handpe": partial(mpjpe, joints="hands", restrict=RestrictType.LOSS),
    # "tlf_handpe": partial(mpjpe, joints="hands", restrict=RestrictType.LOSS_FRAME),
    # "tr_handpe": partial(mpjpe, joints="hands", restrict=RestrictType.REC),
    # "trf_handpe": partial(mpjpe, joints="hands", restrict=RestrictType.REC_FRAME),
    # "tl_lowerpe": partial(mpjpe, joints="lower", restrict=RestrictType.LOSS),
    # "tlf_lowerpe": partial(mpjpe, joints="lower", restrict=RestrictType.LOSS_FRAME),
    # "tr_lowerpe": partial(mpjpe, joints="lower", restrict=RestrictType.REC),
    # "trf_lowerpe": partial(mpjpe, joints="lower", restrict=RestrictType.REC_FRAME),
    # "tl_pred_jitter": partial(
    #     jitter, joints="all", target="pred", restrict=RestrictType.LOSS
    # ),
    # "tlf_pred_jitter": partial(
    #     jitter, joints="all", target="pred", restrict=RestrictType.LOSS_FRAME
    # ),
    # "tr_pred_jitter": partial(
    #     jitter, joints="all", target="pred", restrict=RestrictType.REC
    # ),
    # "trf_pred_jitter": partial(
    #     jitter, joints="all", target="pred", restrict=RestrictType.REC_FRAME
    # ),
    # "trf_pred_H_jitter": partial(
    #     jitter, joints="hands", target="pred", restrict=RestrictType.REC_FRAME
    # ),
    # "trf_pred_LB_jitter": partial(
    #     jitter, joints="lower", target="pred", restrict=RestrictType.REC_FRAME
    # ),
    # "tl_gt_jitter": partial(
    #     jitter, joints="all", target="gt", restrict=RestrictType.LOSS
    # ),
    # "tlf_gt_jitter": partial(
    #     jitter, joints="all", target="gt", restrict=RestrictType.LOSS_FRAME
    # ),
    # "tr_gt_jitter": partial(
    #     jitter, joints="all", target="gt", restrict=RestrictType.REC
    # ),
    # "trf_gt_jitter": partial(
    #     jitter, joints="all", target="gt", restrict=RestrictType.REC_FRAME
    # ),
    # "trf_gt_H_jitter": partial(
    #     jitter, joints="hands", target="gt", restrict=RestrictType.REC_FRAME
    # ),
    # "trf_gt_LB_jitter": partial(
    #     jitter, joints="lower", target="gt", restrict=RestrictType.REC_FRAME
    # ),
}


LOGGING_METRICS = {
    "mpjre",
    "mpjpe",
    "mpjve",
    # "handpe",
    # "upperpe",
    # "lowerpe",
    # "rootpe",
    "pred_jitter",
    "gt_jitter",
    # "trf_pred_jitter",
    # "trf_pred_H_jitter",
}


def get_all_metrics():
    return list(METRIC_FUNCS_DICT.keys())


def get_all_metrics_trackingloss():
    return list(METRIC_FUNCS_DICT_TRACKING_LOSS.keys())


def keep_logging_metrics(all_metrics: dict):
    return {k: v for k, v in all_metrics.items() if k in LOGGING_METRICS}


def get_metric_function(metric):
    if metric in METRIC_FUNCS_DICT_TRACKING_LOSS:
        return METRIC_FUNCS_DICT_TRACKING_LOSS[metric]
    return METRIC_FUNCS_DICT[metric]


def is_array_based_metric(metric):
    return metric in ARRAY_BASED_METRICS


def get_plots_configs():
    return METRICS_PLOTS_CONFIGS
