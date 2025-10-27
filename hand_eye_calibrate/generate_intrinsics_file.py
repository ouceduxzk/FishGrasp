# coding=utf-8
"""
生成相机内参文件

从手眼标定结果中提取相机内参，生成JSON格式的内参文件
"""

import json
import numpy as np
import os
from hand_eye_calibrate import hand_eye_calibrate

def generate_intrinsics_file(output_file="camera_intrinsics.json", gripper_transform=None):
    """
    生成相机内参文件
    
    Args:
        output_file: 输出文件路径
        gripper_transform: 夹爪变换矩阵
    """
    print("=" * 60)
    print("生成相机内参文件")
    print("=" * 60)
    
    # 进行手眼标定获取内参
    print("正在进行手眼标定...")
    R, t, mtx, dist = hand_eye_calibrate(gripper_transform=gripper_transform, enable_error_analysis=True)
    
    # 提取内参
    fx = float(mtx[0, 0])
    fy = float(mtx[1, 1])
    cx = float(mtx[0, 2])
    cy = float(mtx[1, 2])
    
    # 创建内参字典
    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion_coefficients": {
            "k1": float(dist[0, 0]),
            "k2": float(dist[0, 1]),
            "p1": float(dist[0, 2]),
            "p2": float(dist[0, 3]),
            "k3": float(dist[0, 4])
        },
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients_array": dist.tolist(),
        "image_size": {
            "width": 640,
            "height": 480
        },
        "calibration_info": {
            "method": "OpenCV calibrateCamera",
            "reprojection_error": "See pixel_error_analysis.png",
            "note": "Generated from hand-eye calibration"
        }
    }
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(intrinsics, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 内参文件已保存到: {output_file}")
    
    # 打印对比信息
    print("\n" + "=" * 60)
    print("内参对比")
    print("=" * 60)
    
    print("标定得到的内参:")
    print(f"  fx: {fx:.2f}")
    print(f"  fy: {fy:.2f}")
    print(f"  cx: {cx:.2f}")
    print(f"  cy: {cy:.2f}")
    
    print("\n代码中使用的默认内参:")
    print(f"  fx: 615.0")
    print(f"  fy: 615.0")
    print(f"  cx: 320.0")
    print(f"  cy: 240.0")
    
    print("\n误差分析:")
    fx_error = abs(fx - 615.0) / 615.0 * 100
    fy_error = abs(fy - 615.0) / 615.0 * 100
    cx_error = abs(cx - 320.0) / 320.0 * 100
    cy_error = abs(cy - 240.0) / 240.0 * 100
    
    print(f"  fx误差: {fx_error:.1f}%")
    print(f"  fy误差: {fy_error:.1f}%")
    print(f"  cx误差: {cx_error:.1f}%")
    print(f"  cy误差: {cy_error:.1f}%")
    
    # 3D位置误差估算
    print(f"\n" + "=" * 60)
    print("3D位置误差估算")
    print("=" * 60)
    
    # 假设在图像中心的一个点，深度为1米
    test_pixel_x, test_pixel_y = 320, 240
    test_depth = 1.0  # 1米
    
    # 使用标定内参计算3D位置
    x_calibrated = (test_pixel_x - cx) * test_depth / fx
    y_calibrated = (test_pixel_y - cy) * test_depth / fy
    
    # 使用默认内参计算3D位置
    x_default = (test_pixel_x - 320.0) * test_depth / 615.0
    y_default = (test_pixel_y - 240.0) * test_depth / 615.0
    
    # 计算误差
    x_error = abs(x_calibrated - x_default) * 1000  # 转换为毫米
    y_error = abs(y_calibrated - y_default) * 1000  # 转换为毫米
    
    print(f"测试点: 像素({test_pixel_x}, {test_pixel_y}), 深度{test_depth}米")
    print(f"标定内参计算的3D位置: ({x_calibrated:.4f}, {y_calibrated:.4f}, {test_depth})")
    print(f"默认内参计算的3D位置: ({x_default:.4f}, {y_default:.4f}, {test_depth})")
    print(f"X方向误差: {x_error:.1f} 毫米")
    print(f"Y方向误差: {y_error:.1f} 毫米")
    print(f"总位置误差: {np.sqrt(x_error**2 + y_error**2):.1f} 毫米")
    
    return intrinsics

def analyze_3d_error_impact():
    """
    分析内参误差对3D重建的影响
    """
    print(f"\n" + "=" * 60)
    print("3D重建误差影响分析")
    print("=" * 60)
    
    # 标定内参
    fx_cal, fy_cal = 590.76, 586.56
    cx_cal, cy_cal = 340.14, 258.71
    
    # 默认内参
    fx_def, fy_def = 615.0, 615.0
    cx_def, cy_def = 320.0, 240.0
    
    # 测试不同深度和位置的误差
    depths = [0.5, 1.0, 1.5, 2.0]  # 米
    pixel_positions = [
        (160, 120),   # 左上
        (480, 120),   # 右上
        (160, 360),   # 左下
        (480, 360),   # 右下
        (320, 240),   # 中心
    ]
    
    print("不同深度和位置的3D重建误差 (毫米):")
    print("位置(像素)  深度(米)  X误差   Y误差   总误差")
    print("-" * 50)
    
    max_error = 0
    for depth in depths:
        for px, py in pixel_positions:
            # 使用标定内参
            x_cal = (px - cx_cal) * depth / fx_cal
            y_cal = (py - cy_cal) * depth / fy_cal
            
            # 使用默认内参
            x_def = (px - cx_def) * depth / fx_def
            y_def = (py - cy_def) * depth / fy_def
            
            # 计算误差
            x_error = abs(x_cal - x_def) * 1000
            y_error = abs(y_cal - y_def) * 1000
            total_error = np.sqrt(x_error**2 + y_error**2)
            
            max_error = max(max_error, total_error)
            
            print(f"({px:3d},{py:3d})    {depth:4.1f}    {x_error:5.1f}  {y_error:5.1f}   {total_error:5.1f}")
    
    print(f"\n最大3D重建误差: {max_error:.1f} 毫米")
    
    if max_error > 10:
        print("⚠️  警告：3D重建误差较大，建议使用标定内参")
    elif max_error > 5:
        print("⚠️  注意：3D重建误差中等，建议使用标定内参")
    else:
        print("✅ 3D重建误差较小")

def main():
    """主函数"""
    print("相机内参文件生成工具")
    print("这个工具将从手眼标定结果中提取相机内参并生成JSON文件")
    
    # 设置夹爪变换（与手眼标定保持一致）
    gripper_transform = {
        'R': np.array([[1, 0, 0],
                       [0, 1, 0], 
                       [0, 0, 1]]),
        't': np.array([0, 0, 0.195]).reshape(3, 1)
    }
    
    # 生成内参文件
    intrinsics = generate_intrinsics_file("camera_intrinsics.json", gripper_transform)
    
    # 分析3D误差影响
    analyze_3d_error_impact()
    
    print(f"\n" + "=" * 60)
    print("使用建议")
    print("=" * 60)
    print("1. 将生成的 camera_intrinsics.json 复制到项目根目录")
    print("2. 在调用 mask_to_3d.py 时使用 --intrinsics_file camera_intrinsics.json")
    print("3. 在 realtime_segmentation_3d.py 中设置 intrinsics_file 参数")
    print("4. 这将显著提高3D重建的精度")
    
    return intrinsics

if __name__ == "__main__":
    main()
