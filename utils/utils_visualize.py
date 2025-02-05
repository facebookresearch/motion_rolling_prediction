# MIT License
# Copyright (c) 2022 ETH Sensing, Interaction & Perception Lab
#
# This code is based on https://github.com/eth-siplab/AvatarPoser
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import multiprocessing as mp
import os
from functools import partial
from typing import List, Optional

import cv2
import numpy as np
import trimesh
from body_visualizer.mesh.mesh_viewer import MeshViewer

from body_visualizer.tools.vis_tools import colors

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from tqdm import tqdm
from trimesh.creation import cylinder
from trimesh.primitives import Sphere
from trimesh.visual.color import ColorVisuals


#os.environ["PYOPENGL_PLATFORM"] = "egl"
HEIGHT_OFFSET = 0.5  # meters
CAMERA_DIST = 3  # meters
UP_DEFAULT = np.array([0.0, 1.0, 0.0])

AX_LENGTH = 0.15
AX_RADIUS = 0.01
AX_COLORS = [
    colors["red"],
    colors["green"],
    colors["blue"],
]


class CheckerBoard:
    def __init__(self, white=(247, 246, 244), black=(146, 163, 171)):
        self.white = np.array(white) / 255.0
        self.black = np.array(black) / 255.0
        self.verts, self.faces, self.texts = None, None, None
        self.offset = None

    @staticmethod
    def gen_checker_xy(black, white, square_size=0.5, xlength=50.0, ylength=50.0):
        """
        generate a checker board in parallel to x-y plane
        starting from (0, 0) to (xlength, ylength), in meters
        return: trimesh.Trimesh
        """
        xsquares = int(xlength / square_size)
        ysquares = int(ylength / square_size)
        verts, faces, texts = [], [], []
        fcount = 0
        for i in range(xsquares):
            for j in range(ysquares):
                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, j * square_size, 0])
                p3 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])
                p3 = np.array([i * square_size, (j + 1) * square_size, 0])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                if (i + j) % 2 == 0:
                    texts.append(black)
                    texts.append(black)
                else:
                    texts.append(white)
                    texts.append(white)

        # now compose as mesh
        mesh = trimesh.Trimesh(
            vertices=np.array(verts) + np.array([-5, -5, 0]),
            faces=np.array(faces),
            process=False,
            face_colors=np.array(texts),
        )
        return mesh


"""
# --------------------------------
# Visualize avatar using body pose information and body model
# --------------------------------
"""


# SOURCE: https://github.com/Jianghanxiao/Helper3D/blob/master/trimesh_render/src/camera.py#L4 + custom adaptations
def lookAt(eye, target, up):
    # Normalize the up vector
    up /= np.linalg.norm(up)
    forward = eye - target
    forward /= np.linalg.norm(forward)
    if np.dot(forward, up) == 1 or np.dot(forward, up) == -1:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    new_up = np.cross(forward, right)
    new_up /= np.linalg.norm(new_up)

    # Construct a rotation matrix from the right, new_up, and forward vectors
    rotation = np.eye(4)
    rotation[:3, :3] = np.row_stack((right, new_up, forward))

    # Apply a translation to the camera position
    translation = np.eye(4)
    translation[:3, 3] = [
        -np.dot(right, eye),
        -np.dot(new_up, eye),
        -np.dot(forward, eye),
    ]

    camera_pose = np.linalg.inv(np.matmul(translation, rotation))
    return camera_pose


def get_normal_from_body_joints(joints):
    # computes the normal of vectors from hip to both shoulders.
    vec1 = joints[:, 16] - joints[:, 0]
    vec2 = joints[:, 17] - joints[:, 0]
    normal = np.mean(np.cross(vec1, vec2), axis=0)
    normal = normal / np.linalg.norm(normal)
    return normal


def get_camera_pose(joints):
    direction = get_normal_from_body_joints(joints)
    # make sure direction is not below the floor
    point_to_look_at = c2c(joints[:, 0].mean(axis=0))
    eye = point_to_look_at + direction * CAMERA_DIST
    eye[2] = max(1.0 + HEIGHT_OFFSET, eye[2] + HEIGHT_OFFSET)

    camera_matrix = lookAt(
        eye,
        point_to_look_at,
        np.array([0.0, 0.0, 1.0]),
    )
    return camera_matrix


def get_origin_axes_meshes():
    static_meshes = []
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for j in range(3):
        s0 = [0, 0, 0]
        s1 = list(axes[j] * AX_LENGTH * 2)
        points = np.array([s0, s1])
        line = cylinder(radius=AX_RADIUS, segment=points)
        line.visual = ColorVisuals(line, vertex_colors=AX_COLORS[j])
        static_meshes.append(line)
    return static_meshes


def get_rotation_axes_meshes(center, rot):
    static_meshes = []
    for j in range(3):
        if (rot[:, j] == 0).all():
            continue
        s0 = list(center)
        s1 = list(center + rot[:, j] * AX_LENGTH)
        points = np.array([s0, s1])
        line = cylinder(radius=AX_RADIUS, segment=points)
        line.visual = ColorVisuals(line, vertex_colors=AX_COLORS[j])
        static_meshes.append(line)
    return static_meshes


def save_animation(
    body_pose,
    savepath,
    bm,
    fps=60,
    resolution=(800, 800),
    marker_points: Optional[np.ndarray] = None,
    marker_colors: Optional[np.ndarray] = None,
    marker_rot: Optional[np.ndarray] = None,
    show_rot_axes: bool = False,
    show_origin_axes: bool = True,
    export_meshes: bool = False,
) -> List[str]:
    """
    Returns a list of paths to the stored assets (mp4 or meshes obj files)
    """
    imw, imh = resolution
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    faces = c2c(bm.f)

    camera_matrix = get_camera_pose(c2c(body_pose.Jtr))
    mv.update_camera_pose(camera_matrix)

    stored_assets_paths = []
    meshes_path = os.path.join(
        os.path.dirname(savepath), os.path.basename(savepath).split(".")[0]
    )
    if export_meshes:
        os.makedirs(meshes_path, exist_ok=True)
        stored_assets_paths.append(meshes_path)

    img_array = []
    for fId in tqdm(range(body_pose.v.shape[0])):
        # if fId == 300:
        #    break
        try:
            body_mesh = trimesh.Trimesh(
                vertices=c2c(body_pose.v[fId]),
                faces=faces,
                vertex_colors=np.tile([0.9, 0.7, 0.7, 0.8], (6890, 1)),
            )
            if export_meshes:
                path = os.path.join(meshes_path, f"./mesh_{fId:06d}.obj")
                inv_matrix = np.array(
                    [[100, 0, 0, 0], [0, 0, -100, 0], [0, 100, 0, 0], [0, 0, 0, 1]],
                    dtype=np.float32,
                )
                body_mesh.apply_transform(inv_matrix.T)
                body_mesh.export(path, "obj")

            generator = CheckerBoard()
            checker_mesh = generator.gen_checker_xy(generator.black, generator.white)
            static_meshes = [checker_mesh, body_mesh]
            if show_origin_axes:
                static_meshes += get_origin_axes_meshes()

            if marker_points is not None:
                assert marker_colors is not None
                for i in range(marker_points.shape[1]):
                    sphere = Sphere(center=marker_points[fId, i], radius=0.03)
                    sphere.visual = ColorVisuals(
                        sphere, vertex_colors=marker_colors[fId, i]
                    )
                    static_meshes.append(sphere)
                    if marker_rot is not None and show_rot_axes:
                        static_meshes += get_rotation_axes_meshes(
                            marker_points[fId, i], marker_rot[fId, i]
                        )

            mv.set_static_meshes(static_meshes)
            body_image = mv.render(render_wireframe=False)
            body_image = body_image.astype(np.uint8)
            body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)

            img_array.append(body_image)
        except Exception as e:
            print(e)

    out = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution)
    stored_assets_paths.append(savepath)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    return stored_assets_paths
