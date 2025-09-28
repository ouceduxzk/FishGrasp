#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
椭圆核损失训练示例脚本

演示如何使用椭圆核损失训练鱼体关键点检测模型
支持2个关键点：身体中心和头部中心
"""

import os
import sys
import subprocess
from datetime import datetime

def run_ellipsoid_training():
    """运行椭圆核损失训练示例"""
    
    print("="*60)
    print("🐟 椭圆核损失训练示例")
    print("="*60)
    
    # 检查数据目录是否存在
    data_dir = "./process_data"
    annotations_file = "./process_data/train_annotations.json"
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请确保数据目录包含以下结构:")
        print("process_data/")
        print("├── images/")
        print("│   ├── fish1.jpg")
        print("│   └── fish2.jpg")
        print("└── train_annotations.json")
        return
    
    if not os.path.exists(annotations_file):
        print(f"❌ 标注文件不存在: {annotations_file}")
        print("请确保标注文件格式正确:")
        print('{"fish1.jpg": {"landmarks": [[100, 50], [100, 150]], "visibility": [1, 1]}}')
        return
    
    # 生成实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"ellipsoid_{timestamp}"
    
    # 构建训练命令
    cmd = [
        "python3", "train_landmark_model.py",
        "--mode", "train",
        "--data_dir", data_dir,
        "--annotations", annotations_file,
        "--epochs", "50",  # 较少的epochs用于演示
        "--batch_size", "8",
        "--lr", "0.001",
        "--backbone", "resnet18",
        "--exp_name", exp_name,
        "--sharpness", "2.0",  # 较高的锐度用于演示椭圆核效果
        "--test_split", "0.1",
        "--val_split", "0.2"
    ]
    
    print(f"🚀 开始训练...")
    print(f"📁 数据目录: {data_dir}")
    print(f"📄 标注文件: {annotations_file}")
    print(f"🏷️  实验名称: {exp_name}")
    print(f"🔧 椭圆核锐度: 2.0 (高锐度，严格惩罚)")
    print(f"🎯 预测关键点: ['body_center', 'head_center']")
    print()
    
    # 执行训练
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ 训练完成！")
        
        # 显示结果目录
        save_dir = f"experiments/{exp_name}"
        if os.path.exists(save_dir):
            print(f"📁 模型保存在: {save_dir}")
            print(f"📋 训练配置: {save_dir}/training_config.json")
            print(f"🏆 最佳模型: {save_dir}/best_fish_landmark_model_gaussian.pth")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return
    
    # 运行测试（如果模型存在）
    model_path = f"{save_dir}/best_fish_landmark_model_gaussian.pth"
    if os.path.exists(model_path):
        print(f"\n🧪 开始测试模型...")
        test_cmd = [
            "python3", "train_landmark_model.py",
            "--mode", "test",
            "--model_path", model_path,
            "--test_data_dir", data_dir,
            "--test_annotations", annotations_file,
            "--output_dir", f"{save_dir}/test_results"
        ]
        
        try:
            subprocess.run(test_cmd, check=True, capture_output=False)
            print("✅ 测试完成！")
            print(f"📊 测试结果保存在: {save_dir}/test_results")
        except subprocess.CalledProcessError as e:
            print(f"❌ 测试失败: {e}")

def print_ellipsoid_info():
    """打印椭圆核损失信息"""
    print("""
🔬 椭圆核损失特性:

1. 方向感知损失:
   - 椭圆长轴对齐鱼体方向（身体中心 → 头部中心）
   - 短轴垂直于鱼体方向
   - 更符合鱼体几何形状

2. 参数说明:
   - sigma_major: 椭圆长轴标准差 (默认: 0.15)
   - sigma_minor: 椭圆短轴标准差 (默认: 0.05)
   - sharpness: 锐度系数 (值越大越严格)
   - radius: 损失计算半径 (默认: 0.3)

3. 优势:
   - 考虑鱼体方向性
   - 更精确的惩罚机制
   - 适应不同鱼体姿态
   - 提高关键点预测精度

4. 数据格式要求:
   - 2个关键点: [头部中心, 身体中心]
   - JSON格式: {"image.jpg": {"landmarks": [[x1,y1], [x2,y2]], "visibility": [1,1]}}
   - 坐标顺序: 头部中心在前，身体中心在后
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        print_ellipsoid_info()
    else:
        run_ellipsoid_training()
