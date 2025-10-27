# coding=utf-8
"""
相机内参像素误差分析工具

这个脚本提供了详细的相机内参标定误差分析功能，包括：
1. 重投影误差计算
2. 误差统计分析
3. 内参精度评估
4. 可视化图表生成

使用方法：
python pixel_error_analysis.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import cv2
import numpy as np

np.set_printoptions(precision=8, suppress=True)


def calculate_reprojection_errors(obj_points, img_points, mtx, dist, rvecs, tvecs):
    """
    计算重投影误差
    
    Args:
        obj_points: 3D点列表
        img_points: 2D点列表  
        mtx: 相机内参矩阵
        dist: 畸变系数
        rvecs: 旋转向量列表
        tvecs: 平移向量列表
    
    Returns:
        total_error: 总重投影误差
        per_view_errors: 每张图片的误差
        all_errors: 所有点的误差
    """
    total_error = 0
    per_view_errors = []
    all_errors = []
    
    for i in range(len(obj_points)):
        # 重投影3D点到2D
        projected_points, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        projected_points = projected_points.reshape(-1, 2)
        
        # 确保img_points和projected_points的数据类型和形状一致
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


def analyze_intrinsics_accuracy(mtx, dist, obj_points, img_points, rvecs, tvecs):
    """
    分析内参精度
    
    Args:
        mtx: 相机内参矩阵
        dist: 畸变系数
        obj_points: 3D点列表
        img_points: 2D点列表
        rvecs: 旋转向量列表
        tvecs: 平移向量列表
    
    Returns:
        analysis_results: 分析结果字典
    """
    # 计算重投影误差
    total_error, per_view_errors, all_errors = calculate_reprojection_errors(
        obj_points, img_points, mtx, dist, rvecs, tvecs
    )
    
    # 内参分析
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    
    # 计算焦距相对误差（假设标准焦距）
    focal_length_avg = (fx + fy) / 2
    focal_length_ratio = fx / fy
    
    # 主点位置分析
    image_center_x, image_center_y = mtx[0, 2], mtx[1, 2]
    
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
        'intrinsics_analysis': {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'focal_length_avg': focal_length_avg,
            'focal_length_ratio': focal_length_ratio,
            'image_center': (image_center_x, image_center_y)
        },
        'distortion_coefficients': dist.flatten()
    }
    
    return analysis_results


def plot_error_analysis(analysis_results, output_dir="./"):
    """
    绘制误差分析图表
    
    Args:
        analysis_results: 分析结果
        output_dir: 输出目录
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('相机内参标定误差分析', fontsize=16, fontweight='bold')
    
    # 1. 每张图片的重投影误差
    per_view_errors = analysis_results['per_view_errors']
    axes[0, 0].bar(range(len(per_view_errors)), per_view_errors, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('每张图片的重投影误差')
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
    axes[1, 0].boxplot([all_errors], labels=['重投影误差'])
    axes[1, 0].set_title('重投影误差箱线图')
    axes[1, 0].set_ylabel('重投影误差 (像素)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 内参信息表格
    axes[1, 1].axis('off')
    intrinsics = analysis_results['intrinsics_analysis']
    
    table_data = [
        ['参数', '值', '单位'],
        ['fx (X方向焦距)', f'{intrinsics["fx"]:.2f}', '像素'],
        ['fy (Y方向焦距)', f'{intrinsics["fy"]:.2f}', '像素'],
        ['cx (主点X坐标)', f'{intrinsics["cx"]:.2f}', '像素'],
        ['cy (主点Y坐标)', f'{intrinsics["cy"]:.2f}', '像素'],
        ['焦距比例 (fx/fy)', f'{intrinsics["focal_length_ratio"]:.4f}', '无量纲'],
        ['平均焦距', f'{intrinsics["focal_length_avg"]:.2f}', '像素'],
        ['', '', ''],
        ['误差统计', '', ''],
        ['总重投影误差', f'{analysis_results["total_reprojection_error"]:.3f}', '像素'],
        ['误差均值', f'{stats["mean"]:.3f}', '像素'],
        ['误差标准差', f'{stats["std"]:.3f}', '像素'],
        ['最大误差', f'{stats["max"]:.3f}', '像素'],
        ['最小误差', f'{stats["min"]:.3f}', '像素']
    ]
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                            colWidths=[0.4, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif i == 8:  # 分隔行
                cell.set_facecolor('#E0E0E0')
            else:
                cell.set_facecolor('#F5F5F5')
    
    axes[1, 1].set_title('内参和误差统计信息', pad=20)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'camera_calibration_error_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"误差分析图表已保存到: {output_path}")
    
    plt.close()


def load_calibration_data(images_path, arm_pose_file=None):
    """
    加载标定数据
    
    Args:
        images_path: 标定图片路径
        arm_pose_file: 机械臂位姿文件（可选）
    
    Returns:
        obj_points, img_points, mtx, dist, rvecs, tvecs
    """
    print("++++++++++开始加载标定数据++++++++++++++")
    
    # 角点的个数以及棋盘格间距
    XX = 9  # 标定板的中长度对应的角点的个数
    YY = 6  # 标定板的中宽度对应的角点的个数
    L = 0.02475  # 标定板一格的长度  单位为米

    # 设置寻找亚像素角点的参数
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
    objp = L * objp

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    for i in range(0, 20):  # 处理标定图片
        image = f"{images_path}/{i}.jpg"
        print(f"正在处理第{i}张图片：{image}")

        if os.path.exists(image):
            img = cv2.imread(image)
            print(f"图像大小： {img.shape}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            
            if ret:
                print(f"找到角点，左上角：{corners[0, 0]}, 右下角：{corners[-1, -1]}")
                obj_points.append(objp)
                
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                if len(corners2) > 0:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
            else:
                print(f"未找到角点：{image}")

    if len(img_points) == 0:
        raise ValueError("没有找到有效的标定图片！")

    print(f"成功加载 {len(img_points)} 张标定图片")

    # 进行相机标定
    print("++++++++++开始相机标定++++++++++++++")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    
    print("内参矩阵:")
    print(mtx)
    print("畸变系数:")
    print(dist)
    print("++++++++++相机标定完成++++++++++++++")

    return obj_points, img_points, mtx, dist, rvecs, tvecs


def main():
    """
    主函数：执行完整的像素误差分析
    """
    # 设置路径
    images_path = "./collect_data"
    output_dir = "./"
    
    try:
        # 加载标定数据
        obj_points, img_points, mtx, dist, rvecs, tvecs = load_calibration_data(images_path)
        
        # 进行误差分析
        print("\n++++++++++开始像素误差分析++++++++++++++")
        analysis_results = analyze_intrinsics_accuracy(mtx, dist, obj_points, img_points, rvecs, tvecs)
        
        # 打印分析结果
        print(f"总重投影误差: {analysis_results['total_reprojection_error']:.4f} 像素")
        print(f"误差统计:")
        stats = analysis_results['error_statistics']
        print(f"  均值: {stats['mean']:.4f} 像素")
        print(f"  标准差: {stats['std']:.4f} 像素")
        print(f"  最大值: {stats['max']:.4f} 像素")
        print(f"  最小值: {stats['min']:.4f} 像素")
        print(f"  中位数: {stats['median']:.4f} 像素")
        
        print(f"\n内参分析:")
        intrinsics = analysis_results['intrinsics_analysis']
        print(f"  fx: {intrinsics['fx']:.2f} 像素")
        print(f"  fy: {intrinsics['fy']:.2f} 像素")
        print(f"  cx: {intrinsics['cx']:.2f} 像素")
        print(f"  cy: {intrinsics['cy']:.2f} 像素")
        print(f"  焦距比例 (fx/fy): {intrinsics['focal_length_ratio']:.4f}")
        
        # 生成误差分析图表
        plot_error_analysis(analysis_results, output_dir)
        
        print("++++++++++像素误差分析完成++++++++++++++")
        
        # 评估标定质量
        print("\n++++++++++标定质量评估++++++++++++++")
        total_error = analysis_results['total_reprojection_error']
        if total_error < 0.5:
            print("✅ 标定质量：优秀 (重投影误差 < 0.5 像素)")
        elif total_error < 1.0:
            print("✅ 标定质量：良好 (重投影误差 < 1.0 像素)")
        elif total_error < 2.0:
            print("⚠️  标定质量：一般 (重投影误差 < 2.0 像素)")
        else:
            print("❌ 标定质量：较差 (重投影误差 >= 2.0 像素)")
            print("建议：重新采集标定数据或检查标定板质量")
        
        # 焦距比例检查
        focal_ratio = intrinsics['focal_length_ratio']
        if abs(focal_ratio - 1.0) < 0.01:
            print("✅ 焦距比例：正常 (fx/fy ≈ 1.0)")
        else:
            print(f"⚠️  焦距比例：异常 (fx/fy = {focal_ratio:.4f})")
            print("建议：检查相机传感器是否为正方形像素")
        
    except Exception as e:
        print(f"错误：{e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 像素误差分析完成！")
    else:
        print("\n❌ 像素误差分析失败！")
