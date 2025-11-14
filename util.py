import math
import numpy as np
import cv2
from typing import Tuple, Optional


def detect_tail_direction(mask_bool: np.ndarray, vx: float, vy: float, centroid: Tuple[float, float], 
                          debug: bool = False, debug_output_path: Optional[str] = None) -> bool:
    """
    Detect which direction along the principal axis points to the tail.
    
    Strategy: Divide mask into two halves along the SECOND principal axis (perpendicular to main axis),
    and check which half is more uniform/rectangular (tail is more uniform).
    
    Args:
        mask_bool: HxW boolean array; True indicates body pixels
        vx, vy: First principal direction vector (unit vector) - main body axis
        centroid: (cx, cy) centroid of the mask
        debug: If True, save debug visualizations
        debug_output_path: Path prefix for debug output files (e.g., "debug_mask_001")
    
    Returns:
        True if (vx, vy) points to tail, False if (-vx, -vy) points to tail
    """
    ys, xs = np.where(mask_bool)
    if ys.size < 20:
        return True  # Default: assume current direction is correct
    
    cx, cy = centroid
    
    # Get all points
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    pts_centered = pts - np.array([cx, cy])
    
    # Compute PCA to get principal directions for the whole fish
    U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)
    if Vt.shape[0] < 2:
        return True  # Cannot compute principal directions
    
    # First principal direction (main body axis) - should match (vx, vy)
    dir1 = Vt[0]
    # Second principal direction (perpendicular to first)
    dir2 = Vt[1]
    
    # Normalize second principal direction
    norm2 = math.hypot(dir2[0], dir2[1]) or 1.0
    v2x = dir2[0] / norm2
    v2y = dir2[1] / norm2
    
    # Rotate all points to align with principal axis
    angle = math.atan2(dir1[1], dir1[0])
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts_rotated = (R @ pts_centered.T).T
    
    # Fit axis-aligned rectangle (min bounding box) for the whole fish
    x_min, x_max = pts_rotated[:, 0].min(), pts_rotated[:, 0].max()
    y_min, y_max = pts_rotated[:, 1].min(), pts_rotated[:, 1].max()
    
    # Split the rectangle into two halves along the first principal axis (x-axis in rotated space)
    x_mid = (x_min + x_max) / 2.0
    
    # Split points into two halves based on their x-coordinate in rotated space
    half1_mask_rotated = pts_rotated[:, 0] < x_mid
    half2_mask_rotated = pts_rotated[:, 0] >= x_mid
    
    if half1_mask_rotated.sum() < 10 or half2_mask_rotated.sum() < 10:
        return True  # Not enough points, use default
    
    # Get points for each half (in original space)
    half1_pts = pts[half1_mask_rotated]
    half2_pts = pts[half2_mask_rotated]
    
    # Create boolean masks for each half
    h, w = mask_bool.shape
    half1_mask_bool = np.zeros_like(mask_bool)
    half2_mask_bool = np.zeros_like(mask_bool)
    for i, pt in enumerate(pts):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < h and 0 <= x < w:
            if half1_mask_rotated[i]:
                half1_mask_bool[y, x] = True
            else:
                half2_mask_bool[y, x] = True
    
    # Compute standard deviation for bins in each half using the whole bbox (cut into two)
    def compute_std_bins(mask_half_bool, rect_x_min, rect_x_max, R, cx, cy, y_min, y_max, angle, return_rect_info=False):
        """
        Divide bbox into 20 bins along the first principal axis and compute standard deviation
        of pixel counts in each bin.
        
        Args:
            mask_half_bool: HxW boolean mask for this half
            rect_x_min: Minimum x coordinate of the bbox half (in rotated space)
            rect_x_max: Maximum x coordinate of the bbox half (in rotated space)
            R: Rotation matrix
            cx, cy: Centroid coordinates
            y_min, y_max: y range of bbox (in rotated space)
            angle: Rotation angle
            return_rect_info: If True, also return rectangle info for visualization
            
        Returns:
            std_dev or (std_dev, rect_info)
        """
        try:
            # Divide bbox into 10 bins along the first principal axis (x-axis in rotated space)
            num_bins = 10
            bin_width = (rect_x_max - rect_x_min) / num_bins
            bin_edges = np.linspace(rect_x_min, rect_x_max, num_bins + 1)
            
            # Create a hash bool map for fast lookup: check if a pixel is in the mask
            h, w = mask_half_bool.shape
            # Use a set for O(1) lookup: store (x, y) tuples of mask pixels
            mask_pixel_set = set()
            y_coords_mask, x_coords_mask = np.where(mask_half_bool)
            for x, y in zip(x_coords_mask, y_coords_mask):
                mask_pixel_set.add((x, y))
            
            # For each bin, count how many pixels are inside the mask
            bin_pixel_counts = []
            bin_centers = []
            
            # Get the inverse rotation matrix to convert from rotated space back to original
            R_inv = R.T
            
            for i in range(num_bins):
                bin_x_min = bin_edges[i]
                bin_x_max = bin_edges[i + 1]
                bin_center = (bin_x_min + bin_x_max) / 2.0
                bin_centers.append(bin_center)
                
                # Count pixels in this bin
                pixel_count = 0
                # Sample points in this bin (along x and y axes)
                num_x_samples = max(1, int(bin_x_max - bin_x_min) + 1)
                num_y_samples = max(1, int(y_max - y_min) + 1)
                x_samples = np.linspace(bin_x_min, bin_x_max, num_x_samples)
                y_samples = np.linspace(y_min, y_max, num_y_samples)
                
                for x_pos in x_samples:
                    for y_pos in y_samples:
                        # Convert this point from rotated space to original space
                        point_rotated = np.array([x_pos, y_pos])
                        point_centered = (R_inv @ point_rotated) + np.array([cx, cy])
                        
                        # Round to nearest pixel coordinates
                        x_orig = int(np.round(point_centered[0]))
                        y_orig = int(np.round(point_centered[1]))
                        
                        # Check if this pixel is inside the mask using hash lookup
                        if (x_orig, y_orig) in mask_pixel_set:
                            pixel_count += 1
                
                bin_pixel_counts.append(pixel_count)
            
            # Print pixel counts for each bin
            #print(f"  [像素统计] bbox范围: x=[{rect_x_min:.1f}, {rect_x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
            #print(f"  [像素统计] 10个bins的像素数: {bin_pixel_counts}")
            #
            # Compute standard deviation after removing max and min
            if len(bin_pixel_counts) > 2:
                # Remove one max and one min value before computing standard deviation
                bin_pixel_counts_filtered = bin_pixel_counts.copy()
                # Find and remove one occurrence of max
                max_idx = bin_pixel_counts_filtered.index(max(bin_pixel_counts_filtered))
                bin_pixel_counts_filtered.pop(max_idx)
                # Find and remove one occurrence of min
                min_idx = bin_pixel_counts_filtered.index(min(bin_pixel_counts_filtered))
                bin_pixel_counts_filtered.pop(min_idx)
                std_dev = np.std(bin_pixel_counts_filtered)
                avg_count = np.mean(bin_pixel_counts_filtered)
            elif len(bin_pixel_counts) > 0:
                # If only 1 or 2 bins, compute without filtering
                std_dev = np.std(bin_pixel_counts)
                avg_count = np.mean(bin_pixel_counts)
            else:
                std_dev = float('inf')
                avg_count = 0.0
            
            #print(f"  [像素统计] 平均值: {avg_count:.2f}, 标准差: {std_dev:.2f}")
            
            if return_rect_info:
                rect_info = {
                    'angle': angle,
                    'R': R,
                    'x_min': rect_x_min, 'x_max': rect_x_max,
                    'y_min': y_min, 'y_max': y_max,
                    'centroid': np.array([cx, cy]),
                    'std_dev': std_dev,
                    'bin_pixel_counts': bin_pixel_counts,
                    'bin_centers': bin_centers,
                    'bin_edges': bin_edges
                }
                return std_dev, rect_info
            
            return std_dev
            
        except Exception as e:
            # If computation fails, return infinity
            if return_rect_info:
                return float('inf'), None
            return float('inf')
    
    # Compute standard deviations for bins in both halves using the whole bbox (cut into two)
    #print(f"[检测] 开始计算两半的bins标准差...")
    #print(f"[检测] Half1 bbox: x=[{x_min:.1f}, {x_mid:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
    if debug:
        std_dev1, rect_info1 = compute_std_bins(half1_mask_bool, x_min, x_mid, R, cx, cy, y_min, y_max, angle, return_rect_info=True)
    else:
        std_dev1 = compute_std_bins(half1_mask_bool, x_min, x_mid, R, cx, cy, y_min, y_max, angle, return_rect_info=False)
        rect_info1 = None
    
    #print(f"[检测] Half2 bbox: x=[{x_mid:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
    if debug:
        std_dev2, rect_info2 = compute_std_bins(half2_mask_bool, x_mid, x_max, R, cx, cy, y_min, y_max, angle, return_rect_info=True)
    else:
        std_dev2 = compute_std_bins(half2_mask_bool, x_mid, x_max, R, cx, cy, y_min, y_max, angle, return_rect_info=False)
        rect_info2 = None
    
    #print(f"[检测] Half1 标准差: {std_dev1:.2f}, Half2 标准差: {std_dev2:.2f}")
    #print(f"[检测] 尾部: {'Half1' if std_dev1 < std_dev2 else 'Half2'} (标准差更小)")
    
    # The half with lower standard deviation is the tail (more uniform)
    # Use std_dev as fitting_error for consistency with existing code
    fitting_error1 = std_dev1
    fitting_error2 = std_dev2
    
    # The half with lower fitting error (better rectangle fit) is the tail
    # Now we need to determine which direction along the FIRST axis points to that half
    
    # Project centroids of each half onto the first principal axis
    half1_centroid = half1_pts.mean(axis=0)
    half2_centroid = half2_pts.mean(axis=0)
    
    # Compute signed distance from overall centroid along first axis
    half1_centroid_centered = half1_centroid - np.array([cx, cy])
    half2_centroid_centered = half2_centroid - np.array([cx, cy])
    
    proj1_half1 = half1_centroid_centered @ np.array([vx, vy])
    proj1_half2 = half2_centroid_centered @ np.array([vx, vy])
    
    # Debug visualization
    if debug and debug_output_path:
        try:
            # Create debug visualization
            h, w = mask_bool.shape
            debug_vis = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Draw original mask in gray
            debug_vis[mask_bool] = [128, 128, 128]
            
            # Draw half1 in red
            half1_mask_img = np.zeros_like(mask_bool)
            for pt in half1_pts:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= y < h and 0 <= x < w:
                    half1_mask_img[y, x] = True
            debug_vis[half1_mask_img] = [0, 0, 255]  # Red
            
            # Draw half2 in blue
            half2_mask_img = np.zeros_like(mask_bool)
            for pt in half2_pts:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= y < h and 0 <= x < w:
                    half2_mask_img[y, x] = True
            debug_vis[half2_mask_img] = [255, 0, 0]  # Blue
            
            # Draw dividing line (first PCA axis - the actual cut line)
            line_length = max(h, w)
            line_start = (int(cx - vx * line_length), int(cy - vy * line_length))
            line_end = (int(cx + vx * line_length), int(cy + vy * line_length))
            cv2.line(debug_vis, line_start, line_end, (0, 255, 255), 2)  # Cyan dividing line
            
            # Draw bboxes and bin dividers for each half
            def draw_bbox_with_bins(vis_img, rect_info, color, label):
                if rect_info is None:
                    return
                try:
                    R = rect_info['R']
                    x_min, x_max = rect_info['x_min'], rect_info['x_max']
                    y_min, y_max = rect_info['y_min'], rect_info['y_max']
                    centroid = rect_info.get('centroid', np.array([cx, cy]))
                    bin_edges = rect_info.get('bin_edges', [])
                    
                    # Rectangle corners in rotated space
                    corners_rot = np.array([
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max]
                    ], dtype=np.float64)
                    
                    # Rotate back to original space
                    R_inv = R.T  # Inverse rotation
                    corners_orig = (R_inv @ corners_rot.T).T + centroid
                    
                    # Draw rectangle
                    corners_int = corners_orig.astype(np.int32)
                    cv2.polylines(vis_img, [corners_int], True, color, 2)
                    
                    # Draw bin dividers (lines parallel to second principal axis, i.e., y-axis in rotated space)
                    for x_pos in bin_edges:
                        # Line endpoints in rotated space
                        line_pts_rot = np.array([
                            [x_pos, y_min],
                            [x_pos, y_max]
                        ], dtype=np.float64)
                        # Transform to original space
                        line_pts_orig = (R_inv @ line_pts_rot.T).T + centroid
                        line_pts_int = line_pts_orig.astype(np.int32)
                        cv2.line(vis_img, tuple(line_pts_int[0]), tuple(line_pts_int[1]), color, 1)
                    
                    # Add label
                    center = corners_int.mean(axis=0).astype(int)
                    cv2.putText(vis_img, label, tuple(center), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                except Exception as e:
                    pass
            
            # Draw bbox and bin dividers for half1 (red)
            if rect_info1:
                draw_bbox_with_bins(debug_vis, rect_info1, (0, 0, 255), f"H1 (std={fitting_error1:.2f})")
            
            # Draw bbox and bin dividers for half2 (blue)
            if rect_info2:
                draw_bbox_with_bins(debug_vis, rect_info2, (255, 0, 0), f"H2 (std={fitting_error2:.2f})")
            
            # Add text
            tail_half = "Half1" if fitting_error1 < fitting_error2 else "Half2"
            cv2.putText(debug_vis, f"Tail: {tail_half} (lower std)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_vis, f"Std1: {fitting_error1:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(debug_vis, f"Std2: {fitting_error2:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Save debug visualization
            debug_path = f"{debug_output_path}_debug_division.png"
            cv2.imwrite(debug_path, debug_vis)
            
            # Also save individual halves with their bboxes and bin dividers
            half1_vis = np.zeros((h, w, 3), dtype=np.uint8)
            half1_vis[half1_mask_img] = [255, 255, 255]
            if rect_info1:
                draw_bbox_with_bins(half1_vis, rect_info1, (0, 255, 0), f"Std: {fitting_error1:.2f}")
            cv2.imwrite(f"{debug_output_path}_half1.png", half1_vis)
            
            half2_vis = np.zeros((h, w, 3), dtype=np.uint8)
            half2_vis[half2_mask_img] = [255, 255, 255]
            if rect_info2:
                draw_bbox_with_bins(half2_vis, rect_info2, (0, 255, 0), f"Std: {fitting_error2:.2f}")
            cv2.imwrite(f"{debug_output_path}_half2.png", half2_vis)
            
        except Exception as e:
            print(f"[调试] 保存调试可视化失败: {e}")
    
    # Determine which half is tail (lower average = tail)
    if fitting_error1 < fitting_error2:
        # half1 is tail (lower average)
        # Check which direction along first axis points to half1
        if proj1_half1 < 0:
            return True  # Negative direction points to tail, current direction is correct
        else:
            return False  # Positive direction points to tail, need to flip
    else:
        # half2 is tail (lower average)
        # Check which direction along first axis points to half2
        if proj1_half2 < 0:
            return True  # Negative direction points to tail, current direction is correct
        else:
            return False  # Positive direction points to tail, need to flip


def estimate_body_angle_alpha1(mask_bool: np.ndarray, return_details: bool = False, 
                               debug: bool = False, debug_output_path: Optional[str] = None):
    """
    Estimate the principal body direction from a binary mask using SVD (PCA).
    Ensures the direction vector points toward the tail.

    Angle definition (consistent with realtime_segmentation_3d):
    - alpha_1 is the signed angle between the principal axis and the image vertical axis (y-axis).
    - Computed as atan2(vx, vy) where principal direction is (vx, vy) in image coords.
    - Returns radians in [-pi, pi].

    Args:
        mask_bool: HxW boolean array; True indicates body pixels
        return_details: if True, also return (dir_unit, centroid)
        debug: If True, enable debug mode for tail detection
        debug_output_path: Path prefix for debug output files

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

    # Detect tail direction and flip if necessary
    points_to_tail = detect_tail_direction(mask_bool, vx, vy, (float(centroid[0]), float(centroid[1])), 
                                           debug=debug, debug_output_path=debug_output_path)
    if not points_to_tail:
        vx = -vx
        vy = -vy

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


