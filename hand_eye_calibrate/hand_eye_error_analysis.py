#!/usr/bin/env python3
"""
手眼标定像素误差分析工具

这个脚本提供了详细的手眼标定误差分析功能，包括：
1. 手眼标定重投影误差计算
2. 手眼标定精度评估
3. 旋转和平移误差分析
4. 可视化图表生成

使用方法：
python hand_eye_error_analysis.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import cv2
import numpy as np

np.set_printoptions(precision=8, suppress=True)


def euler_angles_to_rotation_matrix(rx, ry, rz):
    """欧拉角转旋转矩阵"""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R


def pose_to_homogeneous_matrix(pose):
    """位姿转齐次矩阵"""
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)
    return R, t


def calculate_hand_eye_reprojection_error(obj_points, img_points, mtx, dist, 
                                        R_arm_list, t_arm_list, R_hand_eye, t_hand_eye):
    """
    计算手眼标定的重投影误差
    
    Args:
        obj_points: 3D点列表
        img_points: 2D点列表
        mtx: 相机内参矩阵
        dist: 畸变系数
        R_arm_list: 机械臂旋转矩阵列表
        t_arm_list: 机械臂平移向量列表
        R_hand_eye: 手眼标定旋转矩阵
        t_hand_eye: 手眼标定平移向量
    
    Returns:
        total_error: 总重投影误差
        per_view_errors: 每张图片的误差
        all_errors: 所有点的误差
    """
    total_error = 0
    per_view_errors = []
    all_errors = []
    
    for i in range(len(obj_points)):
        # 计算相机在世界坐标系下的位姿
        # 手眼标定结果：T_camera_arm = R_hand_eye, t_hand_eye (相机相对于机械臂的变换)
        # 相机在世界坐标系下的位姿：T_world_camera = T_world_arm * T_arm_camera
        # 其中：T_arm_camera = T_camera_arm^(-1)
        
        R_arm = R_arm_list[i]
        t_arm = t_arm_list[i]
        
        # 手眼标定结果：相机相对于机械臂的变换
        R_camera_arm = R_hand_eye
        t_camera_arm = t_hand_eye
        
        # 计算机械臂相对于相机的变换（手眼标定结果的逆）
        R_arm_camera = R_camera_arm.T
        t_arm_camera = -R_camera_arm.T @ t_camera_arm
        
        # 相机在世界坐标系下的位姿
        R_world_camera = R_arm @ R_arm_camera
        t_world_camera = R_arm @ t_arm_camera + t_arm
        
        # 转换为OpenCV格式（旋转向量）
        rvec, _ = cv2.Rodrigues(R_world_camera)
        
        # 重投影3D点到2D
        projected_points, _ = cv2.projectPoints(obj_points[i], rvec, t_world_camera, mtx, dist)
        projected_points = projected_points.reshape(-1, 2)
        
        # 确保数据类型和形状一致
        img_pts = img_points[i].reshape(-1, 2).astype(np.float32)
        proj_pts = projected_points.astype(np.float32)
        
        # 计算误差
        error = cv2.norm(img_pts, proj_pts, cv2.NORM_L2) / len(proj_pts)
        per_view_errors.append(error)
        total_error += error
        
        # 计算每个点的误差
        point_errors = np.sqrt(np.sum((img_pts - proj_pts)**2, axis=1))
        all_errors.extend(point_errors)
    
    total_error /= len(obj_points)
    return total_error, per_view_errors, np.array(all_errors)


def analyze_hand_eye_accuracy(R_hand_eye, t_hand_eye, obj_points, img_points, 
                            mtx, dist, R_arm_list, t_arm_list):
    """
    分析手眼标定精度
    
    Args:
        R_hand_eye: 手眼标定旋转矩阵
        t_hand_eye: 手眼标定平移向量
        obj_points: 3D点列表
        img_points: 2D点列表
        mtx: 相机内参矩阵
        dist: 畸变系数
        R_arm_list: 机械臂旋转矩阵列表
        t_arm_list: 机械臂平移向量列表
    
    Returns:
        analysis_results: 分析结果字典
    """
    # 计算手眼标定重投影误差
    total_error, per_view_errors, all_errors = calculate_hand_eye_reprojection_error(
        obj_points, img_points, mtx, dist, R_arm_list, t_arm_list, R_hand_eye, t_hand_eye
    )
    
    # 分析旋转矩阵
    # 检查旋转矩阵的正交性
    R_orthogonality_error = np.linalg.norm(R_hand_eye @ R_hand_eye.T - np.eye(3))
    
    # 检查行列式（应该接近1）
    det_R = np.linalg.det(R_hand_eye)
    
    # 计算旋转角度
    rotation_angle = np.arccos((np.trace(R_hand_eye) - 1) / 2) * 180 / np.pi
    
    # 分析平移向量
    translation_magnitude = np.linalg.norm(t_hand_eye)
    
    # 计算手眼标定的稳定性（通过不同位姿的误差变化）
    error_std = np.std(per_view_errors)
    error_cv = error_std / np.mean(per_view_errors) if np.mean(per_view_errors) > 0 else 0
    
    analysis_results = {
        'total_reprojection_error': total_error,
        'per_view_errors': per_view_errors,
        'all_point_errors': all_errors,
        'error_statistics': {
            'mean': np.mean(all_errors),
            'std': np.std(all_errors),
            'max': np.max(all_errors),
            'min': np.min(all_errors),
            'median': np.median(all_errors)
        },
        'hand_eye_analysis': {
            'rotation_matrix': R_hand_eye,
            'translation_vector': t_hand_eye,
            'rotation_orthogonality_error': R_orthogonality_error,
            'rotation_determinant': det_R,
            'rotation_angle_degrees': rotation_angle,
            'translation_magnitude': translation_magnitude,
            'error_std': error_std,
            'error_coefficient_of_variation': error_cv
        }
    }
    
    return analysis_results


def plot_hand_eye_error_analysis(analysis_results, output_dir="./"):
    """
    绘制手眼标定误差分析图表
    
    Args:
        analysis_results: 分析结果
        output_dir: 输出目录
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('手眼标定误差分析', fontsize=16, fontweight='bold')
    
    # 1. 每张图片的重投影误差
    per_view_errors = analysis_results['per_view_errors']
    axes[0, 0].bar(range(len(per_view_errors)), per_view_errors, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('每张图片的手眼标定重投影误差')
    axes[0, 0].set_xlabel('图片编号')
    axes[0, 0].set_ylabel('重投影误差 (像素)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加平均线
    mean_error = np.mean(per_view_errors)
    axes[0, 0].axhline(y=mean_error, color='red', linestyle='--', 
                      label=f'平均误差: {mean_error:.3f} 像素')
    axes[0, 0].legend()
    
    # 2. 所有点的误差分布直方图
    all_errors = analysis_results['all_point_errors']
    axes[0, 1].hist(all_errors, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('所有点的重投影误差分布')
    axes[0, 1].set_xlabel('重投影误差 (像素)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加统计信息
    stats = analysis_results['error_statistics']
    axes[0, 1].axvline(x=stats['mean'], color='red', linestyle='--', 
                      label=f'均值: {stats["mean"]:.3f}')
    axes[0, 1].axvline(x=stats['median'], color='orange', linestyle='--', 
                      label=f'中位数: {stats["median"]:.3f}')
    axes[0, 1].legend()
    
    # 3. 误差统计箱线图
    axes[0, 2].boxplot([all_errors], labels=['重投影误差'])
    axes[0, 2].set_title('重投影误差箱线图')
    axes[0, 2].set_ylabel('重投影误差 (像素)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 手眼标定参数表格
    axes[1, 0].axis('off')
    hand_eye = analysis_results['hand_eye_analysis']
    
    table_data = [
        ['参数', '值', '单位/说明'],
        ['总重投影误差', f'{analysis_results["total_reprojection_error"]:.3f}', '像素'],
        ['误差均值', f'{stats["mean"]:.3f}', '像素'],
        ['误差标准差', f'{stats["std"]:.3f}', '像素'],
        ['最大误差', f'{stats["max"]:.3f}', '像素'],
        ['', '', ''],
        ['手眼标定质量', '', ''],
        ['旋转矩阵正交性误差', f'{hand_eye["rotation_orthogonality_error"]:.6f}', '无量纲'],
        ['旋转矩阵行列式', f'{hand_eye["rotation_determinant"]:.6f}', '无量纲'],
        ['旋转角度', f'{hand_eye["rotation_angle_degrees"]:.2f}', '度'],
        ['平移向量模长', f'{hand_eye["translation_magnitude"]:.3f}', '米'],
        ['误差变异系数', f'{hand_eye["error_coefficient_of_variation"]:.3f}', '无量纲']
    ]
    
    table = axes[1, 0].table(cellText=table_data, cellLoc='center', loc='center',
                            colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif i == 5:  # 分隔行
                cell.set_facecolor('#E0E0E0')
            else:
                cell.set_facecolor('#F5F5F5')
    
    axes[1, 0].set_title('手眼标定参数和误差统计', pad=20)
    
    # 5. 旋转矩阵可视化
    axes[1, 1].axis('off')
    R = hand_eye['rotation_matrix']
    
    # 创建旋转矩阵的热力图
    im = axes[1, 1].imshow(R, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_title('手眼标定旋转矩阵')
    
    # 添加数值标注
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f'{R[i, j]:.3f}', 
                           ha='center', va='center', fontsize=10)
    
    # 添加颜色条
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 6. 平移向量可视化
    t_hand_eye = hand_eye['translation_vector']
    axes[1, 2].bar(['X', 'Y', 'Z'], t_hand_eye.flatten(), 
                   color=['red', 'green', 'blue'], alpha=0.7)
    axes[1, 2].set_title('手眼标定平移向量')
    axes[1, 2].set_ylabel('平移量 (米)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 添加数值标注
    for i, v in enumerate(t_hand_eye.flatten()):
        axes[1, 2].text(i, v + 0.01 if v >= 0 else v - 0.01, f'{v:.3f}', 
                       ha='center', va='bottom' if v >= 0 else 'top')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'hand_eye_calibration_error_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"手眼标定误差分析图表已保存到: {output_path}")
    
    plt.close()


def load_hand_eye_calibration_data(images_path, arm_pose_file, gripper_transform=None):
    """
    加载手眼标定数据
    
    Args:
        images_path: 标定图片路径
        arm_pose_file: 机械臂位姿文件
        gripper_transform: 夹爪变换矩阵（可选）
    
    Returns:
        obj_points, img_points, mtx, dist, R_arm_list, t_arm_list
    """
    print("++++++++++开始加载手眼标定数据++++++++++++++")
    
    # 角点的个数以及棋盘格间距
    XX = 9
    YY = 6
    L = 0.02475

    # 设置寻找亚像素角点的参数
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
    objp = L * objp

    obj_points = []
    img_points = []

    for i in range(0, 20):
        image = f"{images_path}/{i}.jpg"
        print(f"正在处理第{i}张图片：{image}")

        if os.path.exists(image):
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                if len(corners2) > 0:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

    if len(img_points) == 0:
        raise ValueError("没有找到有效的标定图片！")

    print(f"成功加载 {len(img_points)} 张标定图片")

    # 进行相机标定
    print("++++++++++开始相机标定++++++++++++++")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print("++++++++++相机标定完成++++++++++++++")

    # 加载机械臂位姿
    print("++++++++++开始加载机械臂位姿++++++++++++++")
    R_arm_list, t_arm_list = process_arm_pose(arm_pose_file, gripper_transform)
    print("++++++++++机械臂位姿加载完成++++++++++++++")

    return obj_points, img_points, mtx, dist, R_arm_list, t_arm_list


def process_arm_pose(arm_pose_file, gripper_transform=None):
    """处理机械臂的pose文件"""
    R_arm, t_arm = [], []
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    for line in all_lines:
        pose = [float(v) for v in line.split(',')]
        pose[0] = pose[0] / 1000  # 转换为米
        pose[1] = pose[1] / 1000
        pose[2] = pose[2] / 1000

        R, t = pose_to_homogeneous_matrix(pose=pose)
        
        # 如果提供了夹爪变换矩阵，进行坐标系转换
        if gripper_transform is not None:
            R_gripper = gripper_transform['R']
            t_gripper = gripper_transform['t']
            
            R = R @ R_gripper
            t = R @ t_gripper + t
            
        R_arm.append(R)
        t_arm.append(t)
    
    return R_arm, t_arm


def perform_hand_eye_calibration(R_arm_list, t_arm_list, rvecs, tvecs):
    """执行手眼标定"""
    print("++++++++++开始手眼标定++++++++++++++")
    R, t = cv2.calibrateHandEye(R_arm_list, t_arm_list, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    print("++++++++++手眼标定完成++++++++++++++")
    return R, t


def main():
    """
    主函数：执行完整的手眼标定误差分析
    """
    # 设置路径
    images_path = "./collect_data"
    arm_pose_file = "./collect_data/poses.txt"
    output_dir = "./"
    
    # 夹爪变换矩阵（如果需要）
    gripper_transform = {
        'R': np.array([[1, 0, 0],
                       [0, 1, 0], 
                       [0, 0, 1]]),
        't': np.array([0, 0, 0.195]).reshape(3, 1)
    }
    
    try:
        # 加载标定数据
        obj_points, img_points, mtx, dist, R_arm_list, t_arm_list = load_hand_eye_calibration_data(
            images_path, arm_pose_file, gripper_transform
        )
        
        # 执行手眼标定
        R_hand_eye, t_hand_eye = perform_hand_eye_calibration(R_arm_list, t_arm_list, 
                                                             [cv2.Rodrigues(R)[0] for R in R_arm_list], 
                                                             t_arm_list)
        
        # 进行手眼标定误差分析
        print("\n++++++++++开始手眼标定误差分析++++++++++++++")
        analysis_results = analyze_hand_eye_accuracy(
            R_hand_eye, t_hand_eye, obj_points, img_points, 
            mtx, dist, R_arm_list, t_arm_list
        )
        
        # 打印分析结果
        print(f"手眼标定总重投影误差: {analysis_results['total_reprojection_error']:.4f} 像素")
        print(f"误差统计:")
        stats = analysis_results['error_statistics']
        print(f"  均值: {stats['mean']:.4f} 像素")
        print(f"  标准差: {stats['std']:.4f} 像素")
        print(f"  最大值: {stats['max']:.4f} 像素")
        print(f"  最小值: {stats['min']:.4f} 像素")
        print(f"  中位数: {stats['median']:.4f} 像素")
        
        print(f"\n手眼标定质量分析:")
        hand_eye = analysis_results['hand_eye_analysis']
        print(f"  旋转矩阵正交性误差: {hand_eye['rotation_orthogonality_error']:.6f}")
        print(f"  旋转矩阵行列式: {hand_eye['rotation_determinant']:.6f}")
        print(f"  旋转角度: {hand_eye['rotation_angle_degrees']:.2f} 度")
        print(f"  平移向量模长: {hand_eye['translation_magnitude']:.3f} 米")
        print(f"  误差变异系数: {hand_eye['error_coefficient_of_variation']:.3f}")
        
        # 生成误差分析图表
        plot_hand_eye_error_analysis(analysis_results, output_dir)
        
        print("++++++++++手眼标定误差分析完成++++++++++++++")
        
        # 评估手眼标定质量
        print("\n++++++++++手眼标定质量评估++++++++++++++")
        total_error = analysis_results['total_reprojection_error']
        if total_error < 1.0:
            print("✅ 手眼标定质量：优秀 (重投影误差 < 1.0 像素)")
        elif total_error < 2.0:
            print("✅ 手眼标定质量：良好 (重投影误差 < 2.0 像素)")
        elif total_error < 5.0:
            print("⚠️  手眼标定质量：一般 (重投影误差 < 5.0 像素)")
        else:
            print("❌ 手眼标定质量：较差 (重投影误差 >= 5.0 像素)")
            print("建议：重新采集标定数据或检查机械臂位姿精度")
        
        # 旋转矩阵质量检查
        orthogonality_error = hand_eye['rotation_orthogonality_error']
        if orthogonality_error < 1e-6:
            print("✅ 旋转矩阵质量：优秀 (正交性误差 < 1e-6)")
        elif orthogonality_error < 1e-4:
            print("✅ 旋转矩阵质量：良好 (正交性误差 < 1e-4)")
        else:
            print("⚠️  旋转矩阵质量：一般 (正交性误差 >= 1e-4)")
        
        # 行列式检查
        det_R = hand_eye['rotation_determinant']
        if abs(det_R - 1.0) < 1e-6:
            print("✅ 旋转矩阵行列式：正常 (det ≈ 1.0)")
        else:
            print(f"⚠️  旋转矩阵行列式：异常 (det = {det_R:.6f})")
        
    except Exception as e:
        print(f"错误：{e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 手眼标定误差分析完成！")
    else:
        print("\n❌ 手眼标定误差分析失败！")
