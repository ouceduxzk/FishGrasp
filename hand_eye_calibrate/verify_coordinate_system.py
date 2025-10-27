#!/usr/bin/env python3
"""
验证手眼标定坐标系定义

这个脚本用于验证手眼标定的坐标系定义是否正确
"""

import os
import cv2
import numpy as np

def verify_coordinate_system():
    """验证坐标系定义"""
    print("=" * 60)
    print("手眼标定坐标系定义验证")
    print("=" * 60)
    
    # 加载数据
    images_path = "./collect_data"
    arm_pose_file = "./collect_data/poses.txt"
    
    # 标定板参数
    XX, YY = 9, 6
    L = 0.02475
    
    # 加载标定数据
    obj_points = []
    img_points = []
    
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
    objp = L * objp
    
    for i in range(20):
        image_path = f"{images_path}/{i}.jpg"
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                img_points.append(corners2)
    
    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    
    # 加载机械臂位姿
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
        pose[0] = pose[0] / 1000
        pose[1] = pose[1] / 1000
        pose[2] = pose[2] / 1000
        
        R, t = pose_to_homogeneous_matrix(pose)
        R_arm_list.append(R)
        t_arm_list.append(t)
    
    # 手眼标定
    rvecs_arm = [cv2.Rodrigues(R)[0] for R in R_arm_list]
    R_hand_eye, t_hand_eye = cv2.calibrateHandEye(R_arm_list, t_arm_list, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    
    print("手眼标定结果:")
    print(f"R_hand_eye:\n{R_hand_eye}")
    print(f"t_hand_eye:\n{t_hand_eye}")
    
    # 验证坐标系定义
    print("\n" + "=" * 60)
    print("坐标系定义验证")
    print("=" * 60)
    
    # 使用第一个位姿进行验证
    i = 0
    R_arm = R_arm_list[i]
    t_arm = t_arm_list[i]
    
    print(f"位姿 {i}:")
    print(f"机械臂位姿 - R_arm:\n{R_arm}")
    print(f"机械臂位姿 - t_arm: {t_arm.flatten()}")
    
    # 原始相机位姿（从相机标定获得）
    R_camera_original, _ = cv2.Rodrigues(rvecs[i])
    t_camera_original = tvecs[i]
    
    print(f"\n原始相机位姿（相机标定结果）:")
    print(f"R_camera_original:\n{R_camera_original}")
    print(f"t_camera_original: {t_camera_original.flatten()}")
    
    # 方法1：直接使用相机标定结果
    print(f"\n方法1 - 直接使用相机标定结果:")
    rvec1, _ = cv2.Rodrigues(R_camera_original)
    projected_points1, _ = cv2.projectPoints(obj_points[i], rvec1, t_camera_original, mtx, dist)
    projected_points1 = projected_points1.reshape(-1, 2)
    
    img_pts = img_points[i].reshape(-1, 2).astype(np.float32)
    proj_pts1 = projected_points1.astype(np.float32)
    error1 = cv2.norm(img_pts, proj_pts1, cv2.NORM_L2) / len(proj_pts1)
    
    print(f"重投影误差: {error1:.4f} 像素")
    print(f"图像点范围: X[{img_pts[:, 0].min():.1f}, {img_pts[:, 0].max():.1f}], Y[{img_pts[:, 1].min():.1f}, {img_pts[:, 1].max():.1f}]")
    print(f"投影点范围: X[{proj_pts1[:, 0].min():.1f}, {proj_pts1[:, 0].max():.1f}], Y[{proj_pts1[:, 1].min():.1f}, {proj_pts1[:, 1].max():.1f}]")
    
    # 方法2：通过手眼标定计算（假设T_camera_arm）
    print(f"\n方法2 - 假设手眼标定返回T_camera_arm:")
    R_camera_arm = R_hand_eye
    t_camera_arm = t_hand_eye
    
    # 计算机械臂相对于相机的变换
    R_arm_camera = R_camera_arm.T
    t_arm_camera = -R_camera_arm.T @ t_camera_arm
    
    # 相机在世界坐标系下的位姿
    R_world_camera2 = R_arm @ R_arm_camera
    t_world_camera2 = R_arm @ t_arm_camera + t_arm
    
    rvec2, _ = cv2.Rodrigues(R_world_camera2)
    projected_points2, _ = cv2.projectPoints(obj_points[i], rvec2, t_world_camera2, mtx, dist)
    projected_points2 = projected_points2.reshape(-1, 2)
    
    proj_pts2 = projected_points2.astype(np.float32)
    error2 = cv2.norm(img_pts, proj_pts2, cv2.NORM_L2) / len(proj_pts2)
    
    print(f"重投影误差: {error2:.4f} 像素")
    print(f"投影点范围: X[{proj_pts2[:, 0].min():.1f}, {proj_pts2[:, 0].max():.1f}], Y[{proj_pts2[:, 1].min():.1f}, {proj_pts2[:, 1].max():.1f}]")
    
    # 方法3：通过手眼标定计算（假设T_arm_camera）
    print(f"\n方法3 - 假设手眼标定返回T_arm_camera:")
    R_arm_camera3 = R_hand_eye
    t_arm_camera3 = t_hand_eye
    
    # 相机在世界坐标系下的位姿
    R_world_camera3 = R_arm @ R_arm_camera3
    t_world_camera3 = R_arm @ t_arm_camera3 + t_arm
    
    rvec3, _ = cv2.Rodrigues(R_world_camera3)
    projected_points3, _ = cv2.projectPoints(obj_points[i], rvec3, t_world_camera3, mtx, dist)
    projected_points3 = projected_points3.reshape(-1, 2)
    
    proj_pts3 = projected_points3.astype(np.float32)
    error3 = cv2.norm(img_pts, proj_pts3, cv2.NORM_L2) / len(proj_pts3)
    
    print(f"重投影误差: {error3:.4f} 像素")
    print(f"投影点范围: X[{proj_pts3[:, 0].min():.1f}, {proj_pts3[:, 0].max():.1f}], Y[{proj_pts3[:, 1].min():.1f}, {proj_pts3[:, 1].max():.1f}]")
    
    # 比较结果
    print(f"\n" + "=" * 60)
    print("结果比较")
    print("=" * 60)
    
    print(f"方法1（直接使用相机标定）: {error1:.4f} 像素")
    print(f"方法2（T_camera_arm）: {error2:.4f} 像素")
    print(f"方法3（T_arm_camera）: {error3:.4f} 像素")
    
    # 找出最佳方法
    errors = [error1, error2, error3]
    methods = ["直接使用相机标定", "T_camera_arm", "T_arm_camera"]
    best_idx = np.argmin(errors)
    
    print(f"\n最佳方法: {methods[best_idx]} (误差: {errors[best_idx]:.4f} 像素)")
    
    # 检查OpenCV文档中的坐标系定义
    print(f"\n" + "=" * 60)
    print("OpenCV坐标系定义检查")
    print("=" * 60)
    
    print("根据OpenCV文档:")
    print("- cv2.calibrateHandEye 求解的是 T_camera_arm")
    print("- 即：相机坐标系相对于机械臂坐标系的变换")
    print("- 公式：T_world_camera = T_world_arm * T_arm_camera")
    print("- 其中：T_arm_camera = T_camera_arm^(-1)")
    
    if best_idx == 1:
        print("\n✅ 验证结果：手眼标定确实返回T_camera_arm")
        print("✅ 坐标系定义正确：方法2是最佳的")
    elif best_idx == 2:
        print("\n⚠️  验证结果：手眼标定可能返回T_arm_camera")
        print("⚠️  坐标系定义需要调整：方法3是最佳的")
    else:
        print("\n❌ 验证结果：手眼标定结果有问题")
        print("❌ 需要检查标定数据质量")

def main():
    """主函数"""
    print("手眼标定坐标系定义验证工具")
    print("这个工具将验证手眼标定的坐标系定义是否正确")
    
    try:
        verify_coordinate_system()
    except Exception as e:
        print(f"验证过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
