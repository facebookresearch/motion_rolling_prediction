# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


from enum import Enum

from typing import Final


NUM_BETAS_SMPL: Final[int] = 16
NUM_JOINTS_SMPL: Final[int] = 22
HEAD_JOINT_IDX: Final[int] = 15
LHAND_JOINT_IDX: Final[int] = 20
RHAND_JOINT_IDX: Final[int] = 21
LFOOT_JOINT_IDX: Final[int] = 10
RFOOT_JOINT_IDX: Final[int] = 11
INPUT_HEAD_IDCES = (36, 37, 38)
INPUT_LHAND_IDCES = (39, 40, 41)
INPUT_RHAND_IDCES = (42, 43, 44)


class DatasetType(str, Enum):
    DEFAULT = "default"
    AMASS = "amass_p1"
    AMASSFULL = "amass_p2"
    ITW = "itw"
    GORP = "gorp"


class MotionLossType(str, Enum):
    LOSS = "LOSS"
    ROT_MSE = "ROT_MSE"
    VEL_MSE = "VEL_MSE"
    JITTER = "JITTER"
    CORRECTION = "CORRECTION"
    JOINTS_MSE = "JOINTS_MSE"
    JOINTS_VEL_MSE = "JOINTS_VEL_MSE"
    JOINTS_VEL_AVG_MSE = "JOINTS_VEL_AVG_MSE"
    JOINTS_VEL_FEET = "JOINTS_VEL_FEET"


class DataTypeGT(str, Enum):
    SPARSE = "SPARSE"
    MOTION_CTX = "MOTION_CTX"
    RELATIVE_ROTS = "RELATIVE_ROTS"
    GLOBAL_ROTS = "GLOBAL_ROTS"
    WORLD_JOINTS = "WORLD_JOINTS"
    SHAPE_PARAMS = "SHAPE_PARAMS"
    SMPL_GENDER = "SMPL_GENDER"
    BODY_PARAMS = "BODY_PARAMS"
    HEAD_MOTION = "HEAD_MOTION"
    TRACKING_GAP = "TRACKING_GAP"
    FILENAME = "FILENAME"
    NUM_FRAMES = "NUM_FRAMES"
    SMPL_MODEL_TYPE = "SMPL_MODEL_TYPE"


class ModelOutputType(str, Enum):
    RELATIVE_ROTS = "RELATIVE_ROTS"
    SHAPE_PARAMS = "SHAPE_PARAMS"
    WORLD_JOINTS = "WORLD_JOINTS"


class DiffusionType(str, Enum):
    STANDARD = "standard"
    ROLLING = "rolling"


class RollingType(str, Enum):
    UNIFORM = "uniform"
    ROLLING = "rolling"


class NoiseScheduleType(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_NOEXP = "cosine_noexp"


class LossDistType(str, Enum):
    L1 = "L1"
    L2 = "L2"


class FreeRunningJumpType(str, Enum):
    NONE = "none"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"


class DiffusionLossWeightType(str, Enum):
    CONSTANT = "constant"
    SNR = "snr"
    INV_SNR = "inv_snr"
    TRUNC_SNR = "trunc_snr"
    VMIN_SNR = "vmin_snr_"
    MIN_SNR = "min_snr_"
    MAX_SNR = "max_snr_"

    @classmethod
    def parse(cls, value):
        for member in cls:
            if value.startswith(member.value):
                suffix = value[len(member.value) :]  # Extract suffix
                if suffix != "":
                    suffix = float(suffix)
                return member, suffix
        raise ValueError(f"{value} is not a valid {cls.__name__} with suffix")


class RollingVisType(str, Enum):
    NONE = "none"
    ROLLING = "rolling"
    ROLLING_CR = "rolling_cr"
    BOXES_CR = "boxes_cr"


class BackpropThroughTimeType(str, Enum):
    NONE = "none"
    FULL = "full"
    BLOCKWISE = "blockwise"
    FULL_PAST = "full_past"
    BLOCKWISE_PAST = "blockwise_past"


class PredictionTargetType(str, Enum):
    POSITIONS = "positions"  # the model output is returned as is
    POSITIONS_SNR = (
        "positions_snr"  # the output of tanh is scaled by the SNR (alpha / sigma)
    )
    POSITIONS_SIGMA = "positions_sigma"  # the output of tanh is scaled by sigma
    POSITIONS_SIGMA2 = (
        "positions_sigma2"  # the output of tanh is scaled by 2*sigma instead of sigma
    )
    POSITIONS_CONST = "positions_const"
    REL_OFFSETS = "rel_offsets"
    ABS_OFFSETS = "abs_offsets"
    PCAF_COSINE = "pcaf_cosine"
    PCAF_COSINE_SQ = "pcaf_cosine_sq"
    PCAF_LINEAR = "pcaf_linear"


class PredictionInputType(str, Enum):
    CLEAN = "clean"  # the input of the model is the previous prediction
    NOISY = "noisy"  # the input of the model is the noisy previous prediction
    NONE = (
        "none"  # the input of the model is no information about the previous prediction
    )
    LAST_GENERATED = (
        "last_generated"  # the input of the model is the last generated pose
    )


ENTITIES_IDCES = {
    DatasetType.AMASS: (
        (
            0,
            1,
            2,
            3,
            4,
            5,
            18,
            19,
            20,
            21,
            22,
            23,
            36,
            37,
            38,
            45,
            46,
            47,
        ),  # head rot 6d, rot vel 6d, trans 3d, trans vel 3d
        (
            6,
            7,
            8,
            9,
            10,
            11,
            24,
            25,
            26,
            27,
            28,
            29,
            39,
            40,
            41,
            48,
            49,
            50,
        ),  # left hand rot 6d, rot vel 6d, trans 3d, trans vel 3d
        (
            12,
            13,
            14,
            15,
            16,
            17,
            30,
            31,
            32,
            33,
            34,
            35,
            42,
            43,
            44,
            51,
            52,
            53,
        ),  # right hand rot 6d, rot vel 6d, trans 3d, trans vel 3d
    ),
}
ENTITIES_IDCES[DatasetType.AMASSFULL] = ENTITIES_IDCES[DatasetType.AMASS]
ENTITIES_IDCES[DatasetType.GORP] = ENTITIES_IDCES[DatasetType.AMASS]
ENTITIES_IDCES[DatasetType.ITW] = ENTITIES_IDCES[DatasetType.AMASS]

TRACKING_IDCES_TO_EXPORT_POS = {
    DatasetType.AMASS: (
        (
            36,
            37,
            38,
        ),  # head position
        (
            39,
            40,
            41,
        ),  # left hand position
        (
            42,
            43,
            44,
        ),  # right hand position
    ),
}
TRACKING_IDCES_TO_EXPORT_POS[DatasetType.AMASSFULL] = TRACKING_IDCES_TO_EXPORT_POS[
    DatasetType.AMASS
]
TRACKING_IDCES_TO_EXPORT_POS[DatasetType.ITW] = TRACKING_IDCES_TO_EXPORT_POS[
    DatasetType.AMASS
]

TRACKING_IDCES_TO_EXPORT_ROT = {
    DatasetType.AMASS: (
        (
            0,
            1,
            2,
            3,
            4,
            5,
        ),  # head 6d rot
        (
            6,
            7,
            8,
            9,
            10,
            11,
        ),  # left hand 6d rot
        (
            12,
            13,
            14,
            15,
            16,
            17,
        ),  # right hand 6d rot
    ),
}
TRACKING_IDCES_TO_EXPORT_ROT[DatasetType.AMASSFULL] = TRACKING_IDCES_TO_EXPORT_ROT[
    DatasetType.AMASS
]
TRACKING_IDCES_TO_EXPORT_ROT[DatasetType.ITW] = TRACKING_IDCES_TO_EXPORT_ROT[
    DatasetType.AMASS
]


ENTITIES_SMPL_IDCES = {
    DatasetType.AMASS: ((15,), (20,), (21,)),  # head, left hand, right hand
}
ENTITIES_SMPL_IDCES[DatasetType.AMASSFULL] = ENTITIES_SMPL_IDCES[DatasetType.AMASS]
ENTITIES_SMPL_IDCES[DatasetType.GORP] = ENTITIES_SMPL_IDCES[DatasetType.AMASS]
ENTITIES_SMPL_IDCES[DatasetType.ITW] = ENTITIES_SMPL_IDCES[DatasetType.AMASS]


class ConditionMasker(str, Enum):
    SEQ_ALL = "default"
    SEQ_IDP = "independent"
    SEQ_HANDS_IDP = "hands_idp"
    SEG_ALL = "seg_all"
    SEG_HANDS = "seg_hands"
    SEG_HANDS_IDP = "seg_hands_idp"

    @classmethod
    def is_seqwise(cls, value):
        return value in [cls.SEQ_ALL, cls.SEQ_IDP, cls.SEQ_HANDS_IDP]

    @classmethod
    def is_idp(cls, value):
        return value in [
            cls.SEQ_IDP,
            cls.SEQ_HANDS_IDP,
            cls.SEG_HANDS_IDP,
        ]

    @classmethod
    def parse(cls, value):
        for member in cls:
            if value == member.value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")

    @classmethod
    def get_entities_idces(cls, value, dataset):
        """
        Return the list of entities to mask.
        """
        idces = ENTITIES_IDCES[dataset]  # headset + both hands
        if value in [
            cls.SEG_HANDS,
            cls.SEQ_HANDS_IDP,
            cls.SEG_HANDS_IDP,
        ]:
            # discard headset (never masked)
            idces = idces[1:]

        if not cls.is_idp(value):
            # join all idces to same entity
            idces = [[elem for sublist in idces for elem in sublist]]

        return idces

    @classmethod
    def get_entities_smpl_idces(cls, value, dataset):
        """
        Return the list of entities to mask.
        """
        idces = ENTITIES_SMPL_IDCES[dataset]  # headset + both hands
        if value in [
            cls.SEG_HANDS,
            cls.SEQ_HANDS_IDP,
            cls.SEG_HANDS_IDP,
        ]:
            # discard headset (never masked)
            idces = idces[1:]

        if not cls.is_idp(value):
            # join all idces to same entity
            idces = [[i for idc in idces for i in idc]]

        return idces


class SMPLGenderParam(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    NEUTRAL = "NEUTRAL"


class SMPLModelType(str, Enum):
    SMPLH = "SMPLH"
    SMPLX = "SMPLX"

    @classmethod
    def parse(cls, value):
        if (
            "smplh" in value.lower()
            or "smpl_h" in value.lower()
            or "smpl-h" in value.lower()
        ):
            return cls.SMPLH
        elif (
            "smplx" in value.lower()
            or "smpl_x" in value.lower()
            or "smpl-x" in value.lower()
        ):
            return cls.SMPLX
        raise ValueError(f"{value} is not a valid {cls.__name__} with suffix")


class TransitionType(str, Enum):
    S_TO_T = "S_TO_T"
    T_TO_S = "T_TO_S"
