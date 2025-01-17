import numpy as np
import torch
from human_body_prior.tools.rotation_tools import rotate_points_xyz
from utils.rotation_conversions import (
    axis_angle_to_quaternion,
    quaternion_raw_multiply,
    quaternion_to_matrix,
)

HEAD_POSITION_OFFSET = torch.tensor(
    [-0.01568759, 0.02414007, 0.09251691], dtype=torch.float32
)
LEFT_HAND_POSITION_OFFSET = torch.tensor(
    [0.12183987, -0.05452635, 0.06948892], dtype=torch.float32
)
RIGHT_HAND_POSITION_OFFSET = torch.tensor(
    [-0.11596493, -0.04973321, 0.06753435], dtype=torch.float32
)


def transform_hmd_from_momentum_to_smpl(inputs_3d: torch.Tensor) -> torch.Tensor:
    """
    Receives a tensor of shape [frames, 3, 7] containing the hmd tracking
    for [head, left_controller, right_controller]. Makes the transformations
    necessary to transform the data from **Momentum to SMPL format**.

    The seven elements of the hmd_tracking are in Momentum format:
    - position: [x, y, z]
    - rotation: [rw, rx, ry, rz]
    """

    global_hmd_pos = inputs_3d[..., :3] / 100
    global_hmd_quat = inputs_3d[..., 3:7]

    # center trajectory in the plane xz so that the hmd is always at the origin at time 0
    # head_origin = global_hmd_pos[0, 0]
    # global_hmd_pos[..., 0] = global_hmd_pos[..., 0] - head_origin[0].unsqueeze(0)
    # global_hmd_pos[..., 2] = global_hmd_pos[..., 2] - head_origin[2].unsqueeze(0)

    # ================== ROTATE SO THAT Z IS THE UP DIRECTION ==================
    # see https://github.com/jakelazaroff/til/blob/main/math/convert-between-3d-coordinate-systems.md

    # Rotate trajectory points clockwise 90 around x axis + flips y
    rotation_matrix1 = torch.tensor(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32
    )
    global_hmd_pos = torch.matmul(global_hmd_pos, rotation_matrix1)  # z = -y & y = z

    # rotate rotations of hmd and controllers: from Y-up to Z-up (only rotation around x axis is needed)
    # the quaternion rotations here are w.r.t. the global coordinate system
    q_rot = axis_angle_to_quaternion(torch.tensor([np.pi / 2, 0, 0]))
    global_hmd_quat = quaternion_raw_multiply(
        q_rot,
        global_hmd_quat.reshape(-1, 4),
    ).reshape(global_hmd_quat.shape)

    # headset -->  rotate twice to go from Blueman to SMPL
    # the quaternion rotations here are w.r.t. the object coordinate system (order of multiplication inverted)
    q_headset_original = global_hmd_quat[:, 0]
    q_rot = axis_angle_to_quaternion(torch.tensor([0, 0, -np.pi / 2]))
    q_headset_smpl = quaternion_raw_multiply(q_headset_original, q_rot)
    q_rot = axis_angle_to_quaternion(torch.tensor([0, -np.pi / 2, 0]))
    q_headset_smpl = quaternion_raw_multiply(q_headset_smpl, q_rot)
    global_hmd_quat[:, 0] = q_headset_smpl

    # left hand -->  rotate 180º around x axis to go from Blueman to SMPL
    q_lhand_original = global_hmd_quat[:, 1]
    q_rot = axis_angle_to_quaternion(torch.tensor([np.pi, 0, 0]))
    q_lhand_smpl = quaternion_raw_multiply(q_lhand_original, q_rot)
    global_hmd_quat[:, 1] = q_lhand_smpl

    return torch.cat([global_hmd_pos, global_hmd_quat], dim=-1)


def transform_controllers_from_gorp_to_smpl(
    inputs_3d: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """
    Receives a tensor of shape [frames, 3, 7] containing the hmd tracking
    for [head, left_controller, right_controller]. Makes the transformations
    necessary to transform the data from **GORP to SMPL format**.

    The seven elements of the hmd_tracking are in GORP format:
    - position: [x, y, z]
    - rotation: [rw, rx, ry, rz]
    """

    global_hmd_pos = inputs_3d[..., :3] * scale
    global_hmd_quat = inputs_3d[..., 3:7]

    # ================== ROTATE SO THAT Z IS THE UP DIRECTION ==================
    # see https://github.com/jakelazaroff/til/blob/main/math/convert-between-3d-coordinate-systems.md
    # Rotate trajectory points clockwise 90 around x axis + flips y
    rotation_matrix1 = torch.tensor(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32
    )
    global_hmd_pos = torch.matmul(global_hmd_pos, rotation_matrix1)  # z = -y & y = z

    # rotate rotations of hmd and controllers: from Y-up to Z-up (only rotation around x axis is needed)
    # the quaternion rotations here are w.r.t. the global coordinate system
    # change chirality --> inverted
    quat = global_hmd_quat
    quat[:, :, [1, 2, 3]] *= -1
    # rot is a 3x3 matrix, we need to apply the rotation to the axes
    rot_to_apply = axis_angle_to_quaternion(
        torch.Tensor([np.pi / 2, 0, 0])
    )  # this rotation fixes the axes to the world coordinate system
    quat = quaternion_raw_multiply(rot_to_apply, quat)

    # now swap the axes in the headset and controller orientations
    # HEADSET --> [0, np.pi, 0]
    rot_to_apply = axis_angle_to_quaternion(torch.Tensor([0, np.pi, 0]))
    quat[:, 0] = quaternion_raw_multiply(quat[:, 0], rot_to_apply)
    # LEFT HAND --> [0, 0, np.pi / 2] + [0, np.pi / 2, 0]
    rot_to_apply = axis_angle_to_quaternion(torch.Tensor([0, 0, np.pi / 2]))
    quat[:, 1] = quaternion_raw_multiply(quat[:, 1], rot_to_apply)
    rot_to_apply = axis_angle_to_quaternion(torch.Tensor([0, np.pi / 2, 0]))
    quat[:, 1] = quaternion_raw_multiply(quat[:, 1], rot_to_apply)
    # RIGHT HAND --> [0, 0, -np.pi / 2]
    rot_to_apply = axis_angle_to_quaternion(torch.Tensor([0, 0, -np.pi / 2]))
    quat[:, 2] = quaternion_raw_multiply(quat[:, 2], rot_to_apply)
    rot_to_apply = axis_angle_to_quaternion(torch.Tensor([0, -np.pi / 2, 0]))
    quat[:, 2] = quaternion_raw_multiply(quat[:, 2], rot_to_apply)
    global_hmd_quat = quat

    # now we correct the offsets between the controllers and the wrist
    # headset
    tracked_rot = quaternion_to_matrix(global_hmd_quat[:, 0])
    global_hmd_pos[:, 0] = global_hmd_pos[:, 0] - torch.matmul(
        tracked_rot, torch.tensor(HEAD_POSITION_OFFSET).float()
    )
    # left hand
    tracked_rot = quaternion_to_matrix(global_hmd_quat[:, 1])
    global_hmd_pos[:, 1] = global_hmd_pos[:, 1] - torch.matmul(
        tracked_rot, torch.tensor(LEFT_HAND_POSITION_OFFSET).float()
    )
    # right hand
    tracked_rot = quaternion_to_matrix(global_hmd_quat[:, 2])
    global_hmd_pos[:, 2] = global_hmd_pos[:, 2] - torch.matmul(
        tracked_rot, torch.tensor(RIGHT_HAND_POSITION_OFFSET).float()
    )

    return torch.cat([global_hmd_pos, global_hmd_quat], dim=-1)


def transform_hand_tracking_from_gorp_to_smpl(
    inputs_3d: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """
    Receives a tensor of shape [frames, 3, 7] containing the hmd tracking
    for [head, left_hand, right_hand]. Makes the transformations
    necessary to transform the data from **GORP to SMPL format**.

    The seven elements of the hmd_tracking are in GORP format:
    - position: [x, y, z]
    - rotation: [rw, rx, ry, rz]
    """

    global_hmd_pos = inputs_3d[..., :3] * scale
    global_hmd_quat = inputs_3d[..., 3:7]

    # ================== ROTATE SO THAT Z IS THE UP DIRECTION ==================
    # see https://github.com/jakelazaroff/til/blob/main/math/convert-between-3d-coordinate-systems.md
    # Rotate trajectory points clockwise 90 around x axis + flips y
    rotation_matrix1 = torch.tensor(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32
    )
    global_hmd_pos = torch.matmul(global_hmd_pos, rotation_matrix1)  # z = -y & y = z

    # rotate rotations of hmd and controllers: from Y-up to Z-up (only rotation around x axis is needed)
    # the quaternion rotations here are w.r.t. the global coordinate system
    # change chirality --> inverted
    quat = global_hmd_quat
    quat[:, :, [1, 2, 3]] *= -1
    # rot is a 3x3 matrix, we need to apply the rotation to the axes
    rot_to_apply = axis_angle_to_quaternion(
        torch.Tensor([np.pi / 2, 0, 0])
    )  # this rotation fixes the axes to the world coordinate system
    quat = quaternion_raw_multiply(rot_to_apply, quat)

    # now swap the axes in the headset and controller orientations
    # HEADSET --> [0, np.pi, 0]
    rot_to_apply = axis_angle_to_quaternion(torch.Tensor([0, np.pi, 0]))
    quat[:, 0] = quaternion_raw_multiply(quat[:, 0], rot_to_apply)
    global_hmd_quat = quat

    # now we correct the offsets between the controllers and the wrist
    # headset
    tracked_rot = quaternion_to_matrix(global_hmd_quat[:, 0])
    global_hmd_pos[:, 0] = global_hmd_pos[:, 0] - torch.matmul(
        tracked_rot, torch.tensor(HEAD_POSITION_OFFSET).float()
    )

    return torch.cat([global_hmd_pos, global_hmd_quat], dim=-1)


def transform_hand_tracking_from_itw_to_smpl(inputs_3d: torch.Tensor) -> torch.Tensor:
    """
    Receives a tensor of shape [frames, 3, 7] containing the hmd tracking
    for [head, left_hand, right_hand]. Makes the transformations
    necessary to transform the data from **GORP to SMPL format**.

    The seven elements of the hmd_tracking are in GORP format:
    - position: [x, y, z]
    - rotation: [rw, rx, ry, rz]
    """

    global_hmd_pos = inputs_3d[..., :3] / 100
    global_hmd_quat = inputs_3d[..., 3:7]

    # center trajectory in the plane xz so that the hmd is always at the origin at time 0
    # head_origin = global_hmd_pos[0, 0]
    # global_hmd_pos[..., 0] = global_hmd_pos[..., 0] - head_origin[0].unsqueeze(0)
    # global_hmd_pos[..., 2] = global_hmd_pos[..., 2] - head_origin[2].unsqueeze(0)

    # ================== ROTATE SO THAT Z IS THE UP DIRECTION ==================
    # see https://github.com/jakelazaroff/til/blob/main/math/convert-between-3d-coordinate-systems.md

    # Rotate trajectory points clockwise 90 around x axis + flips y
    rotation_matrix1 = torch.tensor(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32
    )
    global_hmd_pos = torch.matmul(global_hmd_pos, rotation_matrix1)  # z = -y & y = z

    # rotate rotations of hmd and controllers: from Y-up to Z-up (only rotation around x axis is needed)
    # the quaternion rotations here are w.r.t. the global coordinate system
    q_rot = axis_angle_to_quaternion(torch.tensor([np.pi / 2, 0, 0]))
    global_hmd_quat = quaternion_raw_multiply(
        q_rot,
        global_hmd_quat.reshape(-1, 4),
    ).reshape(global_hmd_quat.shape)

    return torch.cat([global_hmd_pos, global_hmd_quat], dim=-1)
