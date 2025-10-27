#!/usr/bin/env python3
"""
测试手眼标定误差分析功能

这个脚本将测试手眼标定误差分析是否正常工作
"""

import os
import sys
import numpy as np

def test_hand_eye_error_analysis():
    """测试手眼标定误差分析功能"""
    print("=" * 60)
    print("测试手眼标定误差分析功能")
    print("=" * 60)
    
    # 检查必要文件是否存在
    required_files = [
        "./collect_data/poses.txt",
        "./hand_eye_error_analysis.py"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少必要文件: {file_path}")
            return False
        else:
            print(f"✅ 找到文件: {file_path}")
    
    # 检查标定图片
    image_count = 0
    for i in range(20):
        image_path = f"./collect_data/{i}.jpg"
        if os.path.exists(image_path):
            image_count += 1
    
    print(f"✅ 找到 {image_count} 张标定图片")
    
    if image_count < 5:
        print("⚠️  标定图片数量较少，可能影响分析结果")
    
    return True

def test_imports():
    """测试模块导入"""
    print(f"\n" + "=" * 60)
    print("测试模块导入")
    print("=" * 60)
    
    try:
        import cv2
        print("✅ OpenCV 导入成功")
    except ImportError as e:
        print(f"❌ OpenCV 导入失败: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib 导入成功")
    except ImportError as e:
        print(f"❌ Matplotlib 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy 导入成功")
    except ImportError as e:
        print(f"❌ NumPy 导入失败: {e}")
        return False
    
    return True

def test_hand_eye_error_analysis_module():
    """测试手眼标定误差分析模块"""
    print(f"\n" + "=" * 60)
    print("测试手眼标定误差分析模块")
    print("=" * 60)
    
    try:
        # 尝试导入模块
        sys.path.append('.')
        from hand_eye_error_analysis import (
            calculate_hand_eye_reprojection_error,
            analyze_hand_eye_accuracy,
            plot_hand_eye_error_analysis
        )
        print("✅ 手眼标定误差分析模块导入成功")
        
        # 测试函数是否存在
        functions = [
            calculate_hand_eye_reprojection_error,
            analyze_hand_eye_accuracy,
            plot_hand_eye_error_analysis
        ]
        
        for func in functions:
            if callable(func):
                print(f"✅ 函数 {func.__name__} 可用")
            else:
                print(f"❌ 函数 {func.__name__} 不可用")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ 手眼标定误差分析模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试手眼标定误差分析模块时出错: {e}")
        return False

def create_test_data():
    """创建测试数据"""
    print(f"\n" + "=" * 60)
    print("创建测试数据")
    print("=" * 60)
    
    # 创建简单的测试数据
    test_data = {
        'obj_points': [np.random.rand(54, 3).astype(np.float32) for _ in range(5)],
        'img_points': [np.random.rand(54, 1, 2).astype(np.float32) for _ in range(5)],
        'mtx': np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32),
        'dist': np.array([[0.1, 0.2, 0.001, 0.002, 0.3]], dtype=np.float32),
        'R_arm_list': [np.eye(3) for _ in range(5)],
        't_arm_list': [np.array([[0.1], [0.2], [0.3]]) for _ in range(5)],
        'R_hand_eye': np.eye(3),
        't_hand_eye': np.array([[0.05], [0.1], [0.15]])
    }
    
    print("✅ 测试数据创建成功")
    return test_data

def test_analysis_functions(test_data):
    """测试分析函数"""
    print(f"\n" + "=" * 60)
    print("测试分析函数")
    print("=" * 60)
    
    try:
        from hand_eye_error_analysis import (
            calculate_hand_eye_reprojection_error,
            analyze_hand_eye_accuracy
        )
        
        # 测试重投影误差计算
        print("测试重投影误差计算...")
        total_error, per_view_errors, all_errors = calculate_hand_eye_reprojection_error(
            test_data['obj_points'],
            test_data['img_points'],
            test_data['mtx'],
            test_data['dist'],
            test_data['R_arm_list'],
            test_data['t_arm_list'],
            test_data['R_hand_eye'],
            test_data['t_hand_eye']
        )
        
        print(f"✅ 重投影误差计算成功")
        print(f"  总误差: {total_error:.4f} 像素")
        print(f"  每张图片误差: {len(per_view_errors)} 个")
        print(f"  所有点误差: {len(all_errors)} 个")
        
        # 测试精度分析
        print("测试精度分析...")
        analysis_results = analyze_hand_eye_accuracy(
            test_data['R_hand_eye'],
            test_data['t_hand_eye'],
            test_data['obj_points'],
            test_data['img_points'],
            test_data['mtx'],
            test_data['dist'],
            test_data['R_arm_list'],
            test_data['t_arm_list']
        )
        
        print(f"✅ 精度分析成功")
        print(f"  分析结果包含 {len(analysis_results)} 个键")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试分析函数时出错: {e}")
        return False

def main():
    """主函数"""
    print("手眼标定误差分析功能测试工具")
    print("这个工具将测试手眼标定误差分析功能是否正常工作")
    
    # 测试文件存在性
    success1 = test_hand_eye_error_analysis()
    
    # 测试模块导入
    success2 = test_imports()
    
    # 测试手眼标定误差分析模块
    success3 = test_hand_eye_error_analysis_module()
    
    # 创建测试数据
    test_data = create_test_data()
    
    # 测试分析函数
    success4 = test_analysis_functions(test_data)
    
    print(f"\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    if success1:
        print("✅ 文件检查测试通过")
    else:
        print("❌ 文件检查测试失败")
    
    if success2:
        print("✅ 模块导入测试通过")
    else:
        print("❌ 模块导入测试失败")
    
    if success3:
        print("✅ 手眼标定误差分析模块测试通过")
    else:
        print("❌ 手眼标定误差分析模块测试失败")
    
    if success4:
        print("✅ 分析函数测试通过")
    else:
        print("❌ 分析函数测试失败")
    
    if success1 and success2 and success3 and success4:
        print("\n🎉 所有测试通过！手眼标定误差分析功能正常工作。")
        print("\n现在你可以运行以下命令进行手眼标定误差分析：")
        print("python hand_eye_error_analysis.py")
        print("或者")
        print("python hand_eye_calibrate.py")
    else:
        print("\n❌ 部分测试失败，请检查配置。")

if __name__ == "__main__":
    main()
