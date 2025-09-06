#!/usr/bin/env python3
"""
手眼标定调试工具

这个脚本用于调试手眼标定中的问题，找出巨大误差的原因
"""

import os
import cv2
import numpy as np
import json

def debug_hand_eye_calibration():
    """调试手眼标定问题"""
    print("=" * 60)
    print("手眼标定调试分析")
    print("=" * 60)
    
    # 1. 检查标定数据
    print("1. 检查标定数据...")
    images_path = "./collect_data"
    arm_pose_file = "./collect_data/poses.txt"
    
    # 检查文件存在性
    if not os.path.exists(arm_pose_file):
        print(f"❌ 机械臂位姿文件不存在: {arm_pose_file}")
        return
    
    # 检查标定图片
    image_count = 0
    for i in range(20):
        image_path = f"{images_path}/{i}.jpg"
        if os.path.exists(image_path):
            image_count += 1
    
    print(f"✅ 找到 {image_count} 张标定图片")
    
    # 2. 检查机械臂位姿数据
    print("\n2. 检查机械臂位姿数据...")
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"✅ 机械臂位姿数据行数: {len(lines)}")
    
    # 检查前几行数据
    for i, line in enumerate(lines[:3]):
        pose = [float(v) for v in line.split(',')]
        print(f"  位姿 {i}: {pose}")
        
        # 检查单位转换
        pose_mm = pose.copy()
        pose_m = [p/1000 for p in pose[:3]] + pose[3:]
        print(f"    原始(mm): {pose_mm}")
        print(f"    转换(m): {pose_m}")
    
    # 3. 检查相机标定数据
    print("\n3. 检查相机标定数据...")
    
    # 标定板参数
    XX, YY = 9, 6
    L = 0.02475
    
    # 加载标定图片并检测角点
    obj_points = []
    img_points = []
    valid_images = []
    
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
                valid_images.append(i)
    
    print(f"✅ 有效标定图片: {len(obj_points)} 张")
    print(f"✅ 标定图片编号: {valid_images}")
    
    if len(obj_points) == 0:
        print("❌ 没有找到有效的标定图片")
        return
    
    # 4. 执行相机标定
    print("\n4. 执行相机标定...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    
    print(f"✅ 相机标定成功: {ret}")
    print(f"内参矩阵:")
    print(mtx)
    print(f"畸变系数:")
    print(dist)
    
    # 5. 检查相机标定结果
    print("\n5. 检查相机标定结果...")
    
    # 计算重投影误差
    total_error = 0
    for i in range(len(obj_points)):
        projected_points, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        projected_points = projected_points.reshape(-1, 2)
        
        img_pts = img_points[i].reshape(-1, 2).astype(np.float32)
        proj_pts = projected_points.astype(np.float32)
        
        error = cv2.norm(img_pts, proj_pts, cv2.NORM_L2) / len(proj_pts)
        total_error += error
    
    total_error /= len(obj_points)
    print(f"✅ 相机标定重投影误差: {total_error:.4f} 像素")
    
    # 6. 检查机械臂位姿处理
    print("\n6. 检查机械臂位姿处理...")
    
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
    
    # 处理机械臂位姿
    R_arm_list = []
    t_arm_list = []
    
    for i, line in enumerate(lines):
        if i >= len(obj_points):  # 只处理有对应图片的位姿
            break
            
        pose = [float(v) for v in line.split(',')]
        pose[0] = pose[0] / 1000  # 转换为米
        pose[1] = pose[1] / 1000
        pose[2] = pose[2] / 1000
        
        R, t = pose_to_homogeneous_matrix(pose)
        R_arm_list.append(R)
        t_arm_list.append(t)
    
    print(f"✅ 处理了 {len(R_arm_list)} 个机械臂位姿")
    
    # 检查位姿数据
    print("前3个机械臂位姿:")
    for i in range(min(3, len(R_arm_list))):
        print(f"  位姿 {i}:")
        print(f"    旋转矩阵:\n{R_arm_list[i]}")
        print(f"    平移向量: {t_arm_list[i].flatten()}")
    
    # 7. 执行手眼标定
    print("\n7. 执行手眼标定...")
    
    # 转换rvecs和tvecs格式
    rvecs_arm = []
    tvecs_arm = []
    
    for i in range(len(R_arm_list)):
        rvec, _ = cv2.Rodrigues(R_arm_list[i])
        rvecs_arm.append(rvec)
        tvecs_arm.append(t_arm_list[i])
    
    R_hand_eye, t_hand_eye = cv2.calibrateHandEye(
        R_arm_list, t_arm_list, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI
    )
    
    print(f"✅ 手眼标定完成")
    print(f"手眼标定旋转矩阵:")
    print(R_hand_eye)
    print(f"手眼标定平移向量:")
    print(t_hand_eye)
    
    # 8. 检查手眼标定结果
    print("\n8. 检查手眼标定结果...")
    
    # 检查旋转矩阵
    det_R = np.linalg.det(R_hand_eye)
    orthogonality_error = np.linalg.norm(R_hand_eye @ R_hand_eye.T - np.eye(3))
    
    print(f"旋转矩阵行列式: {det_R:.6f}")
    print(f"正交性误差: {orthogonality_error:.6f}")
    
    # 9. 调试重投影误差计算
    print("\n9. 调试重投影误差计算...")
    
    # 使用第一个位姿进行测试
    i = 0
    R_arm = R_arm_list[i]
    t_arm = t_arm_list[i]
    
    print(f"测试位姿 {i}:")
    print(f"机械臂旋转矩阵:\n{R_arm}")
    print(f"机械臂平移向量: {t_arm.flatten()}")
    
    # 手眼标定结果
    R_arm_camera = R_hand_eye
    t_arm_camera = t_hand_eye
    
    print(f"手眼标定旋转矩阵:\n{R_arm_camera}")
    print(f"手眼标定平移向量: {t_arm_camera.flatten()}")
    
    # 计算相机在世界坐标系下的位姿
    R_world_camera = R_arm @ R_arm_camera
    t_world_camera = R_arm @ t_arm_camera + t_arm
    
    print(f"相机世界位姿旋转矩阵:\n{R_world_camera}")
    print(f"相机世界位姿平移向量: {t_world_camera.flatten()}")
    
    # 转换为OpenCV格式
    rvec, _ = cv2.Rodrigues(R_world_camera)
    
    # 重投影
    projected_points, _ = cv2.projectPoints(obj_points[i], rvec, t_world_camera, mtx, dist)
    projected_points = projected_points.reshape(-1, 2)
    
    img_pts = img_points[i].reshape(-1, 2).astype(np.float32)
    proj_pts = projected_points.astype(np.float32)
    
    # 计算误差
    error = cv2.norm(img_pts, proj_pts, cv2.NORM_L2) / len(proj_pts)
    print(f"位姿 {i} 重投影误差: {error:.4f} 像素")
    
    # 检查点坐标范围
    print(f"图像点坐标范围: X[{img_pts[:, 0].min():.1f}, {img_pts[:, 0].max():.1f}], Y[{img_pts[:, 1].min():.1f}, {img_pts[:, 1].max():.1f}]")
    print(f"投影点坐标范围: X[{proj_pts[:, 0].min():.1f}, {proj_pts[:, 0].max():.1f}], Y[{proj_pts[:, 1].min():.1f}, {proj_pts[:, 1].max():.1f}]")
    
    # 10. 检查坐标系定义问题
    print("\n10. 检查坐标系定义问题...")
    
    # 检查手眼标定的坐标系定义
    print("手眼标定坐标系定义检查:")
    print("- cv2.calibrateHandEye 返回的是 T_arm_camera")
    print("- 即：相机相对于机械臂的变换")
    print("- 公式：T_world_camera = T_world_arm * T_arm_camera")
    
    # 验证坐标系转换
    print("\n坐标系转换验证:")
    print("原始相机位姿（从相机标定获得）:")
    print(f"rvec: {rvecs[i].flatten()}")
    print(f"tvec: {tvecs[i].flatten()}")
    
    # 将rvec转换为旋转矩阵
    R_camera_original, _ = cv2.Rodrigues(rvecs[i])
    print(f"原始相机旋转矩阵:\n{R_camera_original}")
    
    # 比较两种方法得到的相机位姿
    print("\n位姿比较:")
    print("方法1 - 直接使用相机标定结果:")
    print(f"旋转矩阵:\n{R_camera_original}")
    print(f"平移向量: {tvecs[i].flatten()}")
    
    print("方法2 - 通过手眼标定计算:")
    print(f"旋转矩阵:\n{R_world_camera}")
    print(f"平移向量: {t_world_camera.flatten()}")
    
    # 计算差异
    R_diff = np.linalg.norm(R_camera_original - R_world_camera)
    t_diff = np.linalg.norm(tvecs[i] - t_world_camera)
    
    print(f"\n位姿差异:")
    print(f"旋转矩阵差异: {R_diff:.6f}")
    print(f"平移向量差异: {t_diff:.6f}")
    
    if R_diff > 0.1 or t_diff > 0.1:
        print("⚠️  位姿差异较大，可能存在坐标系定义问题")
    else:
        print("✅ 位姿差异较小，坐标系定义正确")

def main():
    """主函数"""
    print("手眼标定调试工具")
    print("这个工具将帮助找出手眼标定中的问题")
    
    try:
        debug_hand_eye_calibration()
    except Exception as e:
        print(f"调试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
