import math
import numpy as np
import cv2
from typing import Tuple, Optional


def estimate_body_angle_alpha1(mask_bool: np.ndarray, return_details: bool = False):
    """
    Estimate the principal body direction from a binary mask using SVD (PCA).

    Angle definition (consistent with realtime_segmentation_3d):
    - alpha_1 is the signed angle between the principal axis and the image vertical axis (y-axis).
    - Computed as atan2(vx, vy) where principal direction is (vx, vy) in image coords.
    - Returns radians in [-pi, pi].

    Args:
        mask_bool: HxW boolean array; True indicates body pixels
        return_details: if True, also return (dir_unit, centroid)

    Returns:
        alpha_1 or (alpha_1, dir_unit, centroid)
    """
    ys, xs = np.where(mask_bool)
    if ys.size < 10:
        if return_details:
            return 0.0, (1.0, 0.0), (0.0, 0.0)
        return 0.0

    # Subsample for speed if large
    N = ys.size
    if N > 4000:
        idx = np.random.choice(N, 4000, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid

    # SVD to get principal axis
    U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)
    vx, vy = Vt[0, 0], Vt[0, 1]
    norm = math.hypot(vx, vy) or 1.0
    vx /= norm
    vy /= norm

    alpha_1 = math.atan2(vx, vy)  # angle to vertical
    alpha_1 = (alpha_1 + math.pi) % (2 * math.pi) - math.pi

    if return_details:
        return float(alpha_1), (float(vx), float(vy)), (float(centroid[0]), float(centroid[1]))
    return float(alpha_1)


def draw_principal_axis(
    image_bgr: np.ndarray,
    mask_bool: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    scale: float = 120.0,
) -> np.ndarray:
    """
    Draw the principal axis on the BGR image for visualization.

    Args:
        image_bgr: HxWx3 BGR image
        mask_bool: HxW boolean body mask
        color: BGR color for the axis
        thickness: line thickness
        scale: arrow length in pixels

    Returns:
        Annotated image copy (BGR)
    """
    img = image_bgr.copy()
    alpha_1, (vx, vy), (cx, cy) = estimate_body_angle_alpha1(mask_bool, return_details=True)

    # Draw arrow from centroid along principal axis
    p0 = (int(round(cx)), int(round(cy)))
    p1 = (int(round(cx + vx * scale)), int(round(cy + vy * scale)))
    p2 = (int(round(cx - vx * scale)), int(round(cy - vy * scale)))

    cv2.arrowedLine(img, p0, p1, color, thickness, tipLength=0.25)
    cv2.line(img, p0, p2, color, thickness)

    # Put angle text
    cv2.putText(
        img,
        f"alpha1={np.degrees(alpha_1):.1f}deg",
        (p0[0] + 10, p0[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    return img


def angle_between_2d_from_origin(start_xy, target_xy) -> float:
    """
    Signed angle (radians) between vectors from origin to start_xy and target_xy.
    Returns value in [-pi, pi].

    Args:
        start_xy: (x, y)
        target_xy: (x, y)
    """
    v1 = np.asarray(start_xy, dtype=np.float64).reshape(2)
    v2 = np.asarray(target_xy, dtype=np.float64).reshape(2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    d = a2 - a1
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return float(d)


def rpy_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Convert RPY (radians) to rotation matrix with external rotation order R = Rz @ Ry @ Rx.
    Returns 3x3 float32 matrix.
    """
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=np.float32)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)


def tool_offset_to_base(delta_tool_xyz_mm, tcp_rpy) -> Tuple[float, float, float]:
    """
    Transform a tool-frame offset (mm) into base-frame offset using TCP RPY (radians).
    Returns (dx_base, dy_base, dz_base).
    """
    rx, ry, rz = tcp_rpy
    R_base_tool = rpy_to_rotation_matrix(rx, ry, rz)
    delta_tool = np.asarray(delta_tool_xyz_mm, dtype=np.float32).reshape(3, 1)
    delta_base = (R_base_tool @ delta_tool).reshape(3)
    return (float(delta_base[0]), float(delta_base[1]), float(delta_base[2]))


def apply_hand_eye_transform(points: np.ndarray, hand_eye_transform: Optional[np.ndarray]) -> np.ndarray:
    """
    Apply a 4x4 hand-eye homogeneous transform to Nx3 point array. If transform is None or
    input is empty, returns input.
    """
    if hand_eye_transform is None or points.size == 0:
        return points
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.hstack([points.astype(np.float32), ones])  # (N,4)
    transformed = (hand_eye_transform @ homo.T).T  # (N,4)
    return transformed[:, :3]


def estimate_fish_weight(points_gripper: np.ndarray, volume_factor: float = 1.0) -> float:
    """
    依据夹爪坐标系点云的包围盒体积估算鱼重量（kg）。
    使用形状因子缩放体积，并限制结果到合理范围。
    """
    if points_gripper.size == 0 or len(points_gripper) < 3:
        return 0.0
    min_coords = np.min(points_gripper, axis=0)
    max_coords = np.max(points_gripper, axis=0)
    dimensions = max_coords - min_coords
    volume_m3 = float(np.prod(dimensions))
    shape_factor = 0.6
    effective_volume = volume_m3 * shape_factor
    fish_density = 1000.0
    weight_kg = effective_volume * fish_density * float(volume_factor)
    weight_kg = max(0.1, min(weight_kg, 2.0))
    return weight_kg


