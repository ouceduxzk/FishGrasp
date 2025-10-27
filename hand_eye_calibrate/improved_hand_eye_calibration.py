#!/usr/bin/env python3
"""
改进的手眼标定脚本

使用更好的标定参数和数据质量检查
"""

import os
import cv2
import numpy as np
import json

def improved_hand_eye_calibration():
    """改进的手眼标定"""
    print("=" * 60)
    print("改进的手眼标定")
    print("=" * 60)
    
    # 设置路径
    images_path = "./collect_data"
    arm_pose_file = "./collect_data/poses.txt"
    
    # 标定板参数
    XX, YY = 9, 6
    L = 0.02475
    
    # 更严格的亚像素角点检测参数
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 50, 0.0001)
    
    # 加载标定数据
    obj_points = []
    img_points = []
    valid_images = []
    
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
    objp = L * objp
    
    print("1. 加载和验证标定数据...")
    
    for i in range(20):
        image_path = f"{images_path}/{i}.jpg"
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            if ret:
                # 亚像素角点检测
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # 验证角点质量
                if validate_corners(corners2, size):
                    obj_points.append(objp)
                    img_points.append(corners2)
                    valid_images.append(i)
                    print(f"  ✅ 图片 {i}: 角点质量良好")
                else:
                    print(f"  ❌ 图片 {i}: 角点质量不佳")
            else:
                print(f"  ❌ 图片 {i}: 未检测到角点")
    
    print(f"有效标定图片: {len(obj_points)} 张")
    
    if len(obj_points) < 10:
        print("❌ 有效标定图片数量不足")
        return None, None, None, None
    
    # 相机标定
    print("\n2. 执行相机标定...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    
    # 计算相机标定重投影误差
    camera_error = 0
    for i in range(len(obj_points)):
        projected_points, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        projected_points = projected_points.reshape(-1, 2)
        
        img_pts = img_points[i].reshape(-1, 2).astype(np.float32)
        proj_pts = projected_points.astype(np.float32)
        
        error = cv2.norm(img_pts, proj_pts, cv2.NORM_L2) / len(proj_pts)
        camera_error += error
    
    camera_error /= len(obj_points)
    print(f"相机标定重投影误差: {camera_error:.4f} 像素")
    
    if camera_error > 1.0:
        print("⚠️  相机标定误差较大，建议检查标定数据质量")
    
    # 加载机械臂位姿
    print("\n3. 加载机械臂位姿...")
    
    def euler_angles_to_rotation_matrix(rx, ry, rz):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx
    
    def pose_to_homogeneous_matrix(pose):
        x, y, z, rx, ry, rz = pose
        R = euler_angles_to_rotation_matrix(rx, ry, rz)
        t = np.array([x, y, z]).reshape(3, 1)
        return R, t
    
    R_arm_list = []
    t_arm_list = []
    
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if i >= len(obj_points):
            break
        pose = [float(v) for v in line.split(',')]
        pose[0] = pose[0] / 1000  # 转换为米
        pose[1] = pose[1] / 1000
        pose[2] = pose[2] / 1000
        
        R, t = pose_to_homogeneous_matrix(pose)
        R_arm_list.append(R)
        t_arm_list.append(t)
    
    print(f"加载了 {len(R_arm_list)} 个机械臂位姿")
    
    # 验证机械臂位姿质量
    print("\n4. 验证机械臂位姿质量...")
    
    # 检查位姿变化范围
    positions = np.array([t.flatten() for t in t_arm_list])
    position_ranges = np.max(positions, axis=0) - np.min(positions, axis=0)
    
    print(f"位置变化范围: X={position_ranges[0]:.3f}m, Y={position_ranges[1]:.3f}m, Z={position_ranges[2]:.3f}m")
    
    if np.any(position_ranges < 0.1):
        print("⚠️  机械臂位置变化范围较小，可能影响标定精度")
    
    # 检查旋转变化
    rotation_angles = []
    for R in R_arm_list:
        angle = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi
        rotation_angles.append(angle)
    
    rotation_range = max(rotation_angles) - min(rotation_angles)
    print(f"旋转角度范围: {rotation_range:.1f} 度")
    
    if rotation_range < 30:
        print("⚠️  机械臂旋转角度变化较小，可能影响标定精度")
    
    # 手眼标定
    print("\n5. 执行手眼标定...")
    
    # 转换格式
    rvecs_arm = [cv2.Rodrigues(R)[0] for R in R_arm_list]
    
    # 使用不同的手眼标定方法
    methods = [
        ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
        ("PARK", cv2.CALIB_HAND_EYE_PARK),
        ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
        ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
        ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS)
    ]
    
    best_error = float('inf')
    best_method = None
    best_R = None
    best_t = None
    
    for method_name, method_flag in methods:
        try:
            R, t = cv2.calibrateHandEye(R_arm_list, t_arm_list, rvecs, tvecs, method_flag)
            
            # 计算手眼标定重投影误差
            error = calculate_hand_eye_reprojection_error(
                obj_points, img_points, mtx, dist, R_arm_list, t_arm_list, R, t
            )
            
            print(f"  {method_name}: 重投影误差 = {error:.4f} 像素")
            
            if error < best_error:
                best_error = error
                best_method = method_name
                best_R = R
                best_t = t
                
        except Exception as e:
            print(f"  {method_name}: 失败 - {e}")
    
    print(f"\n最佳方法: {best_method} (误差: {best_error:.4f} 像素)")
    
    # 保存结果
    print("\n6. 保存标定结果...")
    
    results = {
        "hand_eye_calibration": {
            "method": best_method,
            "rotation_matrix": best_R.tolist(),
            "translation_vector": best_t.tolist(),
            "reprojection_error": best_error
        },
        "camera_calibration": {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "reprojection_error": camera_error
        },
        "calibration_info": {
            "valid_images": len(obj_points),
            "image_indices": valid_images,
            "position_range": position_ranges.tolist(),
            "rotation_range": rotation_range
        }
    }
    
    with open("improved_hand_eye_calibration_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✅ 标定结果已保存到: improved_hand_eye_calibration_results.json")
    
    return best_R, best_t, mtx, dist

def validate_corners(corners, image_size):
    """验证角点质量"""
    if corners is None or len(corners) == 0:
        return False
    
    # 检查角点是否在图像范围内
    width, height = image_size
    corners_flat = corners.reshape(-1, 2)
    
    if np.any(corners_flat[:, 0] < 0) or np.any(corners_flat[:, 0] >= width):
        return False
    if np.any(corners_flat[:, 1] < 0) or np.any(corners_flat[:, 1] >= height):
        return False
    
    # 检查角点分布
    x_coords = corners_flat[:, 0]
    y_coords = corners_flat[:, 1]
    
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    
    # 角点应该覆盖图像的一定区域
    if x_range < width * 0.3 or y_range < height * 0.3:
        return False
    
    return True

def calculate_hand_eye_reprojection_error(obj_points, img_points, mtx, dist, R_arm_list, t_arm_list, R_hand_eye, t_hand_eye):
    """计算手眼标定重投影误差"""
    total_error = 0
    
    for i in range(len(obj_points)):
        R_arm = R_arm_list[i]
        t_arm = t_arm_list[i]
        
        # 手眼标定结果：相机相对于机械臂的变换
        R_camera_arm = R_hand_eye
        t_camera_arm = t_hand_eye
        
        # 计算机械臂相对于相机的变换
        R_arm_camera = R_camera_arm.T
        t_arm_camera = -R_camera_arm.T @ t_camera_arm
        
        # 相机在世界坐标系下的位姿
        R_world_camera = R_arm @ R_arm_camera
        t_world_camera = R_arm @ t_arm_camera + t_arm
        
        # 转换为OpenCV格式
        rvec, _ = cv2.Rodrigues(R_world_camera)
        
        # 重投影
        projected_points, _ = cv2.projectPoints(obj_points[i], rvec, t_world_camera, mtx, dist)
        projected_points = projected_points.reshape(-1, 2)
        
        img_pts = img_points[i].reshape(-1, 2).astype(np.float32)
        proj_pts = projected_points.astype(np.float32)
        
        error = cv2.norm(img_pts, proj_pts, cv2.NORM_L2) / len(proj_pts)
        total_error += error
    
    return total_error / len(obj_points)

def main():
    """主函数"""
    print("改进的手眼标定工具")
    print("使用更好的标定参数和数据质量检查")
    
    try:
        R, t, mtx, dist = improved_hand_eye_calibration()
        
        if R is not None:
            print("\n🎉 改进的手眼标定完成！")
            print(f"旋转矩阵:\n{R}")
            print(f"平移向量:\n{t}")
        else:
            print("\n❌ 手眼标定失败")
            
    except Exception as e:
        print(f"标定过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()