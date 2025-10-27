# coding=utf-8
"""
测试像素误差分析功能

这个脚本用于测试像素误差分析功能是否正常工作
"""

import os
import sys
import numpy as np

def test_error_analysis_functions():
    """测试误差分析函数"""
    print("开始测试误差分析函数...")
    
    try:
        # 导入分析函数
        from pixel_error_analysis import (
            calculate_reprojection_errors,
            analyze_intrinsics_accuracy,
            plot_error_analysis
        )
        print("✅ 成功导入误差分析函数")
        
        # 创建模拟数据
        print("创建模拟标定数据...")
        
        # 模拟内参矩阵
        mtx = np.array([
            [615.0, 0, 320.0],
            [0, 614.0, 240.0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 模拟畸变系数
        dist = np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # 模拟3D点
        obj_points = []
        img_points = []
        rvecs = []
        tvecs = []
        
        for i in range(5):  # 模拟5张图片
            # 创建3D点（9x6棋盘格）
            objp = np.zeros((54, 3), np.float32)
            objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.02475
            obj_points.append(objp)
            
            # 创建2D点（添加一些噪声）
            imgp = np.random.rand(54, 1, 2) * 100 + 200
            img_points.append(imgp.astype(np.float32))
            
            # 创建旋转和平移向量
            rvec = np.random.rand(3, 1) * 0.1
            tvec = np.array([[0], [0], [0.5]], dtype=np.float32)
            rvecs.append(rvec)
            tvecs.append(tvec)
        
        print("✅ 模拟数据创建完成")
        
        # 测试重投影误差计算
        print("测试重投影误差计算...")
        total_error, per_view_errors, all_errors = calculate_reprojection_errors(
            obj_points, img_points, mtx, dist, rvecs, tvecs
        )
        print(f"✅ 重投影误差计算完成，总误差: {total_error:.4f} 像素")
        
        # 测试内参精度分析
        print("测试内参精度分析...")
        analysis_results = analyze_intrinsics_accuracy(
            mtx, dist, obj_points, img_points, rvecs, tvecs
        )
        print("✅ 内参精度分析完成")
        
        # 打印分析结果
        print(f"总重投影误差: {analysis_results['total_reprojection_error']:.4f} 像素")
        stats = analysis_results['error_statistics']
        print(f"误差统计 - 均值: {stats['mean']:.4f}, 标准差: {stats['std']:.4f}")
        
        # 测试图表生成
        print("测试图表生成...")
        plot_error_analysis(analysis_results, "./")
        print("✅ 图表生成完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_main_script():
    """测试主脚本功能"""
    print("\n开始测试主脚本功能...")
    
    try:
        # 检查是否存在标定数据
        images_path = "./collect_data"
        if not os.path.exists(images_path):
            print(f"⚠️  标定数据目录不存在: {images_path}")
            print("请确保标定数据目录存在并包含标定图片")
            return False
        
        # 检查是否有标定图片
        image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
        if len(image_files) == 0:
            print("⚠️  未找到标定图片文件")
            print("请确保 collect_data 目录中包含 .jpg 格式的标定图片")
            return False
        
        print(f"✅ 找到 {len(image_files)} 张标定图片")
        
        # 检查是否有位姿文件
        poses_file = os.path.join(images_path, "poses.txt")
        if not os.path.exists(poses_file):
            print("⚠️  未找到位姿文件 poses.txt")
            print("像素误差分析不需要位姿文件，但手眼标定需要")
        else:
            print("✅ 找到位姿文件 poses.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 50)
    print("相机内参像素误差分析功能测试")
    print("=" * 50)
    
    # 测试函数功能
    function_test_passed = test_error_analysis_functions()
    
    # 测试主脚本
    script_test_passed = test_main_script()
    
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    
    if function_test_passed:
        print("✅ 误差分析函数测试通过")
    else:
        print("❌ 误差分析函数测试失败")
    
    if script_test_passed:
        print("✅ 主脚本环境测试通过")
    else:
        print("❌ 主脚本环境测试失败")
    
    if function_test_passed and script_test_passed:
        print("\n🎉 所有测试通过！像素误差分析功能可以正常使用。")
        print("\n使用方法:")
        print("1. 独立使用: python pixel_error_analysis.py")
        print("2. 集成使用: 在 hand_eye_calibrate.py 中启用误差分析")
    else:
        print("\n❌ 部分测试失败，请检查环境配置。")
    
    return function_test_passed and script_test_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
