# coding=utf-8
"""
眼在手上 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂末端坐标系的 旋转矩阵和平移向量
A2^{-1}*A1*X=X*B2*B1^{−1}

支持夹爪变换：如果机械臂末端有固定夹爪，需要提供夹爪相对于机械臂末端的变换矩阵


+++++++++++手眼标定完成+++++++++++++++
旋转矩阵：
[[-0.99462885  0.07149648  0.07484454]
 [-0.06962775 -0.9971997   0.02728984]
 [ 0.07658608  0.021932    0.99682173]]
平移向量：
[[ 0.0247092 ]
 [ 0.09912939]
 [-0.25357213]]
 
"""

import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import cv2
import numpy as np

np.set_printoptions(precision=8, suppress=True)

iamges_path = "./collect_data"  # 手眼标定采集的标定版图片所在路径
arm_pose_file = "./collect_data/poses.txt"  # 采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应

# 夹爪变换矩阵（如果机械臂末端有固定夹爪）
# 这是夹爪坐标系相对于机械臂末端坐标系的变换矩阵
# 如果不需要夹爪变换，设置为None
#gripper_transform = None

# 示例：如果夹爪相对于机械臂末端有固定的偏移和旋转，可以这样设置：
# 对于沿末端自然延伸19.5cm的长气动夹爪装置：
gripper_transform = {
    'R': np.array([[1, 0, 0],
                   [0, 1, 0], 
                   [0, 0, 1]]),  # 旋转矩阵（无旋转，保持与末端相同方向）
    't': np.array([0, 0, 0.195]).reshape(3, 1)  # 平移向量，沿Z轴延伸20cm
}


def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
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
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    return R, t


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


def camera_calibrate(iamges_path, enable_error_analysis=True):
    print("++++++++++开始相机标定++++++++++++++")
    # 角点的个数以及棋盘格间距
    XX = 9  # 标定板的中长度对应的角点的个数
    YY = 6  # 标定板的中宽度对应的角点的个数
    L = 0.0243  # 标定板一格的长度  单位为米

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L * objp

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    for i in range(0, 30):  # 标定好的图片在iamges_path路径下，从0.jpg到x.jpg   建议采集30-50张图片以获得更好的畸变系数

        image = f"{iamges_path}/{i}.jpg"
        print(f"正在处理第{i}张图片：{image}")

        if os.path.exists(image):

            img = cv2.imread(image)
            print(f"图像大小： {img.shape}")
            # h_init, width_init = img.shape[:2]
            # img = cv2.resize(src=img, dsize=(width_init // 2, h_init // 2))
            # print(f"图像大小(resize)： {img.shape}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            # print(corners)
            print(f"左上角点：{corners[0, 0]}")
            print(f"右下角点：{corners[-1, -1]}")

            # 绘制角点并显示图像
            cv2.drawChessboardCorners(img, (XX, YY), corners, ret)
            cv2.imshow('Chessboard', img)

            cv2.waitKey(3000)  ## 停留1s, 观察找到的角点是否正确

            if ret:

                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

    N = len(img_points)

    # 标定得到图案在相机坐标系下的位姿
    # 使用畸变系数进行更精确的标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print(f"tvecs {tvecs}")

    # print("ret:", ret)
    print("内参矩阵:\n", mtx)  # 内参数矩阵
    print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    # 进行像素误差分析
    if enable_error_analysis:
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
        plot_error_analysis(analysis_results, iamges_path)
        
        print("++++++++++像素误差分析完成++++++++++++++")

    print("++++++++++相机标定完成++++++++++++++")

    return rvecs, tvecs, mtx, dist


def process_arm_pose(arm_pose_file, gripper_transform=None):
    """处理机械臂的pose文件。 采集数据时， 每行保存一个机械臂的pose信息， 该pose与拍摄的图片是对应的。
    pose信息用6个数标识， 【x,y,z,Rx, Ry, Rz】. 需要把这个pose信息用旋转矩阵表示。
    
    如果提供了夹爪变换矩阵，会将机械臂末端位姿转换到夹爪坐标系。
    """
    
    R_arm, t_arm = [], []
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        all_lines = f.readlines()
    
    for line in all_lines:
        pose = [float(v) for v in line.split(',')]
        print(f"机械臂位姿：{pose}")
        pose[0] = pose[0] /1000
        pose[1] = pose[1] /1000
        pose[2] = pose[2] /1000
        print(f"new 机械臂位姿：{pose}")

        R, t = pose_to_homogeneous_matrix(pose=pose)
        
        # 如果提供了夹爪变换矩阵，进行坐标系转换
        if gripper_transform is not None:
            R_gripper = gripper_transform['R']
            t_gripper = gripper_transform['t']
            
            # 计算夹爪在世界坐标系下的位姿
            # T_world_gripper = T_world_arm * T_arm_gripper
            R = R @ R_gripper
            t = R @ t_gripper + t
            
        R_arm.append(R)
        t_arm.append(t)
    
    return R_arm, t_arm


def load_calibration_data_for_hand_eye():
    """
    为手眼标定误差分析加载标定数据
    """
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
        image = f"{iamges_path}/{i}.jpg"
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

    return obj_points, img_points


def hand_eye_calibrate(gripper_transform=None, enable_error_analysis=True):
    """
    手眼标定主函数
    
    注意：畸变系数在相机标定阶段已经使用，手眼标定使用的是
    已经考虑了畸变校正的rvecs和tvecs（通过cv2.calibrateCamera获得）
    """
    rvecs, tvecs, mtx, dist = camera_calibrate(iamges_path=iamges_path, enable_error_analysis=enable_error_analysis)
    R_arm, t_arm = process_arm_pose(arm_pose_file=arm_pose_file, gripper_transform=gripper_transform)

    # 手眼标定：计算相机坐标系到机器人末端坐标系的变换
    # 注意：rvecs和tvecs已经包含了畸变校正的影响
    R, t = cv2.calibrateHandEye(R_arm, t_arm, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    
    print("+++++++++++手眼标定完成+++++++++++++++")
    print("注意：手眼标定结果已经考虑了相机畸变校正")
    
    # 如果启用了误差分析，进行手眼标定误差分析
    if enable_error_analysis:
        print("\n++++++++++开始手眼标定误差分析++++++++++++++")
        try:
            # 导入手眼标定误差分析函数
            from hand_eye_error_analysis import analyze_hand_eye_accuracy, plot_hand_eye_error_analysis
            
            # 重新加载标定数据用于误差分析
            obj_points, img_points = load_calibration_data_for_hand_eye()
            
            # 进行手眼标定误差分析
            analysis_results = analyze_hand_eye_accuracy(
                R, t, obj_points, img_points, mtx, dist, R_arm, t_arm
            )
            
            # 打印手眼标定误差分析结果
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
            
            # 生成手眼标定误差分析图表
            plot_hand_eye_error_analysis(analysis_results, iamges_path)
            
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
            
            print("++++++++++手眼标定误差分析完成++++++++++++++")
            
        except ImportError:
            print("⚠️  无法导入手眼标定误差分析模块，跳过误差分析")
        except Exception as e:
            print(f"⚠️  手眼标定误差分析失败: {e}")
    
    return R, t, mtx, dist


if __name__ == "__main__":
    # 使用夹爪变换进行手眼标定，启用像素误差分析
    R, t, mtx, dist = hand_eye_calibrate(gripper_transform=gripper_transform, enable_error_analysis=True)

    print("\n手眼标定结果:")
    print("旋转矩阵：")
    print(R)
    print("平移向量：")
    print(t)
    
    print("\n相机内参:")
    print("内参矩阵:")
    print(mtx)
    print("畸变系数:")
    print(dist)
    
    # 如果使用了夹爪变换，输出说明
    if gripper_transform is not None:
        print("\n注意：此标定结果表示相机坐标系相对于夹爪坐标系的变换关系")
    else:
        print("\n注意：此标定结果表示相机坐标系相对于机械臂末端坐标系的变换关系")
