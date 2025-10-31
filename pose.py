import numpy as np
from typing import Tuple, Optional, Dict, Any

from point_clodu_utils import calculate_surface_normal
from util import tool_offset_to_base


def smooth_rpy_transition(current_rpy, target_rpy, max_change: float = 0.1):
    current_rpy = np.array(current_rpy)
    target_rpy = np.array(target_rpy)
    diff = target_rpy - current_rpy
    for i in range(3):
        if diff[i] > np.pi:
            diff[i] -= 2 * np.pi
        elif diff[i] < -np.pi:
            diff[i] += 2 * np.pi
    for i in range(3):
        if abs(diff[i]) > max_change:
            diff[i] = np.sign(diff[i]) * max_change
    return current_rpy + diff


def normal_to_rpy(normal_vector, current_rpy=None):
    z_target = np.asarray(normal_vector, dtype=np.float64)
    z_target = z_target / (np.linalg.norm(z_target) + 1e-9)

    x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    z_new = z_target
    x_new = x_ref - np.dot(x_ref, z_new) * z_new
    x_new = x_new / (np.linalg.norm(x_new) + 1e-9)
    y_new = np.cross(z_new, x_new)
    y_new = y_new / (np.linalg.norm(y_new) + 1e-9)

    R = np.column_stack([x_new, y_new, z_new])
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    target_rpy = np.array([rx, ry, rz], dtype=np.float64)
    if current_rpy is not None:
        target_rpy = smooth_rpy_transition(current_rpy, target_rpy)
    return target_rpy


def calculate_grasp_pose_with_normal(points_gripper: np.ndarray, current_tcp):
    if points_gripper.size == 0 or len(points_gripper) < 3:
        return current_tcp, None

    normal, centroid = calculate_surface_normal(points_gripper, method='pca')
    current_rpy = np.asarray(current_tcp[3:6], dtype=np.float64)
    target_rpy = normal_to_rpy(normal, current_rpy)

    delta_tool_mm = [float(centroid[0] * 1000.0), float(centroid[1] * 1000.0), float(centroid[2] * 1000.0)]
    dx, dy, dz = tool_offset_to_base(delta_tool_mm, current_rpy)

    grasp_pose = np.array([
        dx,
        dy,
        dz - 25.0,
        target_rpy[0],
        target_rpy[1],
        target_rpy[2],
    ], dtype=np.float64)

    normal_info = {
        'centroid': centroid,
        'normal': normal,
        'current_rpy': current_rpy,
        'target_rpy': target_rpy,
        'rpy_change': target_rpy - current_rpy,
    }
    return grasp_pose, normal_info


