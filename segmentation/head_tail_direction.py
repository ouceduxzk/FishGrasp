#!/usr/bin/env python3
"""
Generate head and tail direction of fish from mask images.

Input: mask images from ./mask_data
Output: visualization with arrows pointing to tail direction in ./head_tail_output
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path to import util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from util import estimate_body_angle_alpha1, draw_principal_axis, calculate_fish_grasp_point


def process_mask_file(mask_path: str, output_dir: str, debug: bool = False):
    """
    Process a single mask file and generate visualization with arrow pointing to tail.
    
    Args:
        mask_path: Path to mask image file
        output_dir: Output directory for visualization
        debug: If True, save debug visualizations (divided masks and fitted rectangles)
    """
    # Read mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[错误] 无法读取mask文件: {mask_path}")
        return
    
    # Convert to boolean mask (threshold at 128)
    mask_bool = mask > 128
    
    # Check if mask has enough pixels
    if mask_bool.sum() < 10:
        print(f"[跳过] mask像素太少: {mask_path}")
        return
    
    # Estimate body angle and direction (with tail detection)
    try:
        base_name = Path(mask_path).stem
        debug_path = os.path.join(output_dir, f"{base_name}") if debug else None
        alpha_1, (vx, vy), (cx, cy) = estimate_body_angle_alpha1(
            mask_bool, return_details=True, debug=debug, debug_output_path=debug_path
        )
    except Exception as e:
        print(f"[错误] 计算PCA方向失败 {mask_path}: {e}")
        return
    
    # Calculate grasp point in debug mode
    grasp_point = None
    if debug:
        try:
            grasp_point = calculate_fish_grasp_point(
                mask_bool, vx, vy, (cx, cy), debug=False, debug_output_path=None
            )
            if grasp_point is not None:
                # Create grasp point visualization
                grasp_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                gx, gy = int(np.round(grasp_point[0])), int(np.round(grasp_point[1]))
                
                # Draw mask in gray
                grasp_vis[mask_bool] = [128, 128, 128]
                
                # Draw principal axis
                vis_image_temp = draw_principal_axis(grasp_vis.copy(), mask_bool, color=(0, 255, 0), thickness=2, scale=150.0)
                grasp_vis = vis_image_temp
                
                # Recalculate the intersection lines for visualization
                # We need to compute the same logic as in calculate_fish_grasp_point
                import math
                ys, xs = np.where(mask_bool)
                pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
                pts_centered = pts - np.array([cx, cy])
                
                # Compute PCA to get principal directions
                U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)
                if Vt.shape[0] >= 2:
                    dir1 = Vt[0]
                    dir2 = Vt[1]
                    
                    # Rotate all points to align with principal axis
                    angle = math.atan2(dir1[1], dir1[0])
                    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
                    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    pts_rotated = (R @ pts_centered.T).T
                    
                    # Fit axis-aligned rectangle
                    x_min, x_max = pts_rotated[:, 0].min(), pts_rotated[:, 0].max()
                    y_min, y_max = pts_rotated[:, 1].min(), pts_rotated[:, 1].max()
                    x_mid = (x_min + x_max) / 2.0
                    
                    # Split points into two halves
                    half1_mask_rotated = pts_rotated[:, 0] < x_mid
                    half2_mask_rotated = pts_rotated[:, 0] >= x_mid
                    
                    if half1_mask_rotated.sum() >= 10 and half2_mask_rotated.sum() >= 10:
                        # Get points for each half
                        half1_pts = pts[half1_mask_rotated]
                        half2_pts = pts[half2_mask_rotated]
                        
                        # Determine which half is tail (simplified - use area as proxy)
                        # Actually we need to compute std_dev, but for visualization we can use a simpler approach
                        # For now, let's assume the half with more points in the tail direction is the tail
                        half1_centroid = half1_pts.mean(axis=0)
                        half2_centroid = half2_pts.mean(axis=0)
                        proj1_half1 = (half1_centroid - np.array([cx, cy])) @ np.array([vx, vy])
                        proj1_half2 = (half2_centroid - np.array([cx, cy])) @ np.array([vx, vy])
                        
                        # The half further in the tail direction (positive projection) is likely the tail
                        if proj1_half1 > proj1_half2:
                            tail_pts = half1_pts
                            head_pts = half2_pts
                        else:
                            tail_pts = half2_pts
                            head_pts = half1_pts
                        
                        # Recalculate PCA for head part
                        head_centroid = head_pts.mean(axis=0)
                        head_pts_centered = head_pts - head_centroid
                        U_head, S_head, Vt_head = np.linalg.svd(head_pts_centered, full_matrices=False)
                        
                        # Recalculate PCA for tail part
                        tail_centroid = tail_pts.mean(axis=0)
                        tail_pts_centered = tail_pts - tail_centroid
                        U_tail, S_tail, Vt_tail = np.linalg.svd(tail_pts_centered, full_matrices=False)
                        
                        if Vt_head.shape[0] >= 1 and Vt_tail.shape[0] >= 1:
                            # Head's principal direction
                            head_dir = Vt_head[0]
                            head_dir_norm = math.hypot(head_dir[0], head_dir[1]) or 1.0
                            head_vx = head_dir[0] / head_dir_norm
                            head_vy = head_dir[1] / head_dir_norm
                            
                            # Ensure direction is similar to the whole fish first PCA direction (vx, vy) which points to tail
                            # Check alignment with whole fish PCA direction
                            alignment_with_whole = (head_vx * vx + head_vy * vy)
                            
                            # If the dot product is negative, the directions are opposite, so flip head PCA
                            if alignment_with_whole < 0:
                                head_vx = -head_vx
                                head_vy = -head_vy
                            
                            tail_dir = Vt_tail[0]
                            tail_dir_norm = math.hypot(tail_dir[0], tail_dir[1]) or 1.0
                            tail_vx = tail_dir[0] / tail_dir_norm
                            tail_vy = tail_dir[1] / tail_dir_norm
                            
                            # Ensure tail direction points away from body center
                            tail_centroid_vec = tail_centroid - np.array([cx, cy])
                            if tail_centroid_vec @ np.array([tail_vx, tail_vy]) < 0:
                                tail_vx = -tail_vx
                                tail_vy = -tail_vy
                            
                            # Get second principal direction (perpendicular to first)
                            dir2 = Vt[1]
                            norm2 = math.hypot(dir2[0], dir2[1]) or 1.0
                            v2x = dir2[0] / norm2
                            v2y = dir2[1] / norm2
                            
                            # Ensure second PCA direction points toward head
                            # Project head centroid onto second PCA direction to determine correct orientation
                            head_centroid_vec = head_centroid - np.array([cx, cy])
                            proj_on_second_pca = head_centroid_vec @ np.array([v2x, v2y])
                            
                            # If projection is negative, flip the direction to point toward head
                            if proj_on_second_pca < 0:
                                v2x = -v2x
                                v2y = -v2y
                            
                            # Calculate average of whole fish first PCA direction and head PCA direction
                            # Both should point in similar direction (toward tail)
                            # Whole fish first PCA direction is (vx, vy) - points to tail
                            # Head PCA direction is (head_vx, head_vy) - should also point to tail (after alignment)
                            # The average should be between these two directions
                            avg_vx = (vx + head_vx) / 2.0
                            avg_vy = (vy + head_vy) / 2.0
                            avg_norm = math.hypot(avg_vx, avg_vy) or 1.0
                            avg_vx /= avg_norm
                            avg_vy /= avg_norm
                            
                            # Use head centroid as the point for the average PCA axis
                            avg_axis_point = head_centroid
                            
                            # Get border line (second PCA direction)
                            R_inv = R.T
                            border_pt1_rot = np.array([x_mid, y_min])
                            border_pt2_rot = np.array([x_mid, y_max])
                            border_pt1_orig = (R_inv @ border_pt1_rot) + np.array([cx, cy])
                            border_pt2_orig = (R_inv @ border_pt2_rot) + np.array([cx, cy])
                            
                            h, w = mask_bool.shape
                            
                            # Draw whole fish first PCA axis (green arrow) - for reference
                            whole_line_length = max(h, w) * 0.5
                            line_start_whole = (int(cx - vx * whole_line_length), int(cy - vy * whole_line_length))
                            line_end_whole = (int(cx + vx * whole_line_length), int(cy + vy * whole_line_length))
                            cv2.line(grasp_vis, line_start_whole, line_end_whole, (0, 255, 0), 2)  # Green for whole fish PCA
                            cv2.arrowedLine(grasp_vis, (int(cx), int(cy)), line_end_whole, (0, 255, 0), 2, tipLength=0.2)  # Green arrow
                            
                            # Draw average PCA axis (average of whole fish PCA and head PCA) - Cyan
                            line_length_main = max(h, w) * 0.6
                            avg_cx, avg_cy = avg_axis_point
                            line_start_main = (int(avg_cx - avg_vx * line_length_main), int(avg_cy - avg_vy * line_length_main))
                            line_end_main = (int(avg_cx + avg_vx * line_length_main), int(avg_cy + avg_vy * line_length_main))
                            cv2.line(grasp_vis, line_start_main, line_end_main, (255, 255, 0), 3)  # Cyan for average PCA axis
                            
                            # Also draw head PCA axis for reference (thinner, with arrow)
                            head_cx, head_cy = head_centroid
                            head_line_length = line_length_main * 0.5
                            line_start_head = (int(head_cx - head_vx * head_line_length), int(head_cy - head_vy * head_line_length))
                            line_end_head = (int(head_cx + head_vx * head_line_length), int(head_cy + head_vy * head_line_length))
                            # Draw line
                            cv2.line(grasp_vis, line_start_head, line_end_head, (0, 255, 255), 2)  # Yellow for head PCA (reference)
                            # Draw arrow pointing in head direction
                            cv2.arrowedLine(grasp_vis, (int(head_cx), int(head_cy)), line_end_head, (0, 255, 255), 2, tipLength=0.2)  # Yellow arrow
                            
                            # Draw border line (second PCA direction) - Magenta
                            border_pt1_int = (int(border_pt1_orig[0]), int(border_pt1_orig[1]))
                            border_pt2_int = (int(border_pt2_orig[0]), int(border_pt2_orig[1]))
                            cv2.line(grasp_vis, border_pt1_int, border_pt2_int, (255, 0, 255), 3)  # Magenta
                            
                            # Add labels
                            # Label for whole fish PCA axis
                            axis_mid_whole = ((line_start_whole[0] + line_end_whole[0]) // 2, 
                                             (line_start_whole[1] + line_end_whole[1]) // 2)
                            cv2.putText(grasp_vis, "Whole PCA", 
                                       (axis_mid_whole[0] + 10, axis_mid_whole[1] - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Label for average PCA axis
                            axis_mid_main = ((line_start_main[0] + line_end_main[0]) // 2, 
                                            (line_start_main[1] + line_end_main[1]) // 2)
                            cv2.putText(grasp_vis, "Avg PCA (Whole+Head)", 
                                       (axis_mid_main[0] + 10, axis_mid_main[1] - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                            
                            # Label for head PCA axis
                            axis_mid_head = ((line_start_head[0] + line_end_head[0]) // 2, 
                                             (line_start_head[1] + line_end_head[1]) // 2)
                            cv2.putText(grasp_vis, "Head PCA", 
                                       (axis_mid_head[0] + 10, axis_mid_head[1] - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                            border_mid = ((border_pt1_int[0] + border_pt2_int[0]) // 2, 
                                         (border_pt1_int[1] + border_pt2_int[1]) // 2)
                            cv2.putText(grasp_vis, "Border (2nd PCA)", 
                                       (border_mid[0] + 10, border_mid[1]), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Draw grasp point with multiple visual indicators
                # Large filled circle
                cv2.circle(grasp_vis, (gx, gy), 8, (0, 0, 255), -1)  # Red filled circle
                # Outer circle outline
                cv2.circle(grasp_vis, (gx, gy), 15, (0, 0, 255), 2)  # Red circle outline
                # Cross mark for precise location
                cross_size = 12
                cv2.line(grasp_vis, (gx - cross_size, gy), (gx + cross_size, gy), (255, 255, 255), 2)  # White horizontal line
                cv2.line(grasp_vis, (gx, gy - cross_size), (gx, gy + cross_size), (255, 255, 255), 2)  # White vertical line
                
                # Add text label
                cv2.putText(grasp_vis, "Grasp Point", (gx + 20, gy - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(grasp_vis, f"({gx}, {gy})", (gx + 20, gy + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(grasp_vis, f"File: {base_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save grasp point visualization
                grasp_output_path = os.path.join(output_dir, f"{base_name}_grasp_point.png")
                cv2.imwrite(grasp_output_path, grasp_vis)
                print(f"[调试] 抓取点已保存: {grasp_output_path} ({gx}, {gy})")
        except Exception as e:
            print(f"[调试] 计算抓取点失败: {e}")
            import traceback
            traceback.print_exc()
    
    # Create visualization
    # Convert mask to BGR for visualization
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Draw principal axis with arrow pointing to tail
    vis_image = draw_principal_axis(mask_bgr, mask_bool, color=(0, 255, 0), thickness=3, scale=150.0)
    
    # Add text information
    cv2.putText(vis_image, f"File: {base_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_image, f"Angle: {np.degrees(alpha_1):.1f}deg", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_image, "Arrow -> Tail", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save visualization
    output_path = os.path.join(output_dir, f"{base_name}_direction.png")
    cv2.imwrite(output_path, vis_image)
    print(f"[保存] {output_path}")
    
    # Also save a text file with direction information
    info_path = os.path.join(output_dir, f"{base_name}_direction.txt")
    with open(info_path, 'w') as f:
        f.write(f"Mask file: {mask_path}\n")
        f.write(f"Direction vector (vx, vy): ({vx:.6f}, {vy:.6f})\n")
        f.write(f"Centroid (cx, cy): ({cx:.2f}, {cy:.2f})\n")
        f.write(f"Angle alpha_1 (rad): {alpha_1:.6f}\n")
        f.write(f"Angle alpha_1 (deg): {np.degrees(alpha_1):.2f}\n")
        f.write(f"Arrow points to: TAIL\n")


def main():
    """Main function to process all mask files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process mask files to detect head/tail direction')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode: save divided masks and fitted rectangles')
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    mask_data_dir = script_dir / "mask_data"
    output_dir = script_dir / "head_tail_output"
    
    # Check if mask_data directory exists
    if not mask_data_dir.exists():
        print(f"[错误] mask_data目录不存在: {mask_data_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"[信息] 输出目录: {output_dir}")
    
    # Find all mask image files
    mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    mask_files = []
    for ext in mask_extensions:
        mask_files.extend(list(mask_data_dir.glob(f"*{ext}")))
        mask_files.extend(list(mask_data_dir.glob(f"*{ext.upper()}")))
    
    if len(mask_files) == 0:
        print(f"[错误] 在 {mask_data_dir} 中未找到mask图像文件")
        return
    
    print(f"[信息] 找到 {len(mask_files)} 个mask文件")
    
    # Process each mask file
    success_count = 0
    for mask_file in sorted(mask_files):
        try:
            process_mask_file(str(mask_file), str(output_dir), debug=args.debug)
            success_count += 1
        except Exception as e:
            print(f"[错误] 处理文件失败 {mask_file}: {e}")
    
    print(f"\n[完成] 成功处理 {success_count}/{len(mask_files)} 个文件")
    print(f"[输出] 结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
