# (c) Meta Platforms, Inc. All Rights Reserved

import random
from functools import partial

import torch
from utils.constants import ConditionMasker
from collections.abc import Callable


def create_masker(
    masker: ConditionMasker,
    dataset: str,
    cond_mask_prob: float,
    min_f: int = 0,
    max_f: int = 1,
)->Callable:
    idces = ConditionMasker.get_entities_idces(masker, dataset)
    if ConditionMasker.is_seqwise(masker):
        return partial(
            mask_cond_seqwise_by_idces,
            idces=idces,
            cond_mask_prob=cond_mask_prob,
        )
    return partial(
        mask_cond_segwise_by_idces,
        idces=idces,
        cond_mask_prob=cond_mask_prob,
        min_f=min_f,
        max_f=max_f,
    )


def mask_cond_seqwise_by_idces(
    cond: torch.Tensor,
    training: bool,
    idces: list,
    force_mask: bool = False,
    cond_mask_prob: float = 0.0,
    **kwargs
):
    """
    Mask the sparse embedding independently for each subset inside list of idces (indices).
    For example, idces = [[0], [1, 2], [3, 4, 5]] will mask each subset independently.
    For example, idces = [[0, 1, 2, 4, 5]] will mask all 0-5 as a whole.
    """
    cond = cond.clone()
    bs, n, c = cond.shape
    if force_mask:
        return torch.zeros_like(cond)
    elif training and cond_mask_prob > 0.0:
        mask = torch.bernoulli(
            torch.ones((bs, len(idces)), device=cond.device) * cond_mask_prob
        ).view(
            bs, 1, len(idces), 1
        )  # 1-> use null_cond, 0-> use real cond
        for i, idc in enumerate(idces):
            cond[:, :, idc] = cond[:, :, idc] * (1.0 - mask[:, :, i])
        return cond
    else:
        return cond


def mask_cond_segwise_by_idces(
    cond: torch.Tensor,
    training: bool,
    idces: list,
    force_mask: bool = False,
    cond_mask_prob: float = 0.0,
    min_f: int = 0,
    max_f: int = 1,
    **kwargs
):
    """
    Mask the sparse embedding independently for each subset inside list of idces (indices)
    (same logic as in mask_cond_seqwise_by_idces).
    However, here we mask segments of length min_f to max_f instead of the whole sequence.
    """
    assert max_f >= min_f, "max_f must be greater than or equal to min_f"
    cond = cond.clone()
    bs, n, c = cond.shape
    if force_mask:
        return torch.zeros_like(cond)
    elif training and cond_mask_prob > 0.0 and max_f > min_f:
        for i in range(bs):
            for entity_idces in idces:
                if random.random() < cond_mask_prob:
                    # we mask a segment of random length from min_f to max_f
                    length = random.randint(
                        min_f, min(max_f, n)
                    )  # can't be longer than the whole sequence
                    t0 = random.randint(0, n - length)
                    cond[i, t0 : t0 + length, entity_idces] = 0
        return cond
    else:
        return cond


def compute_masked_segments(
    seq_len: int,
    prob: float = 0.0,
    min_dist: int = 1,
    min_f: int = 0,
    max_f: int = 1,
    left_padding: int = 1,
):
    """
    Masks a sequence of length seq_len with probability prob for each segment.
    """
    c_idx = left_padding  # at least 'left_padding' frames are not masked
    segments = []
    while c_idx < seq_len - min_f:
        if random.random() < prob:
            # mask the segment
            length = random.randint(min_f, max_f + 1)
            if c_idx + length > seq_len:
                length = seq_len - c_idx
            segments.append((c_idx, c_idx + length))
            c_idx += length + min_dist
        else:
            # do not mask the segment
            c_idx += 1
    return segments
