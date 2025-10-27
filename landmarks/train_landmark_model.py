#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鱼体关键点检测模型训练脚本

使用示例:
    # 训练模式
    python train_landmark_model.py --mode train --data_dir ./data --annotations train_annotations.json --epochs 100
    
    # 测试模式
    python train_landmark_model.py --mode test --model_path ./models/best_model.pth --test_data_dir ./test_data --test_annotations test_annotations.json
    
    # 查看帮助
    python train_landmark_model.py --help

功能特性:
    - 支持训练和测试两种模式
    - 支持多种模型架构 (ResNet18, EfficientNet)
    - 自动数据分割 (训练/验证/测试)
    - 支持JSON和TXT格式的标注文件
    - 自动保存训练配置和模型检查点
    - 提供详细的训练和测试统计信息
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt issues
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fish_landmark_detector import FishLandmarkDetector, FishLandmarkDataset, create_data_transforms
from data_loader import FishLandmarkDataLoader


def train_model(data_dir: str, annotations_file: str, epochs: int = 100, 
                batch_size: int = 16, lr: float = 0.001, backbone: str = 'resnet18',
                save_dir: str = 'models', test_split: float = 0.2, val_split: float = 0.2,
                same_folder_mode: bool = False, sharpness: float = 1.0, 
                loss_type: str = 'ellipsoid'):
    """训练鱼体关键点检测模型"""
    
    print("="*60)
    print("🐟 鱼体关键点检测模型训练")
    print("="*60)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    print("📁 加载数据...")
    if same_folder_mode:
        print("📂 使用同文件夹模式：图像和JSON文件在同一目录")
    loader = FishLandmarkDataLoader(data_dir, same_folder_mode=same_folder_mode)
    
    # 根据文件扩展名确定格式
    if annotations_file.endswith('.json'):
        image_paths, landmarks_list = loader.load_from_json(annotations_file)
    elif annotations_file.endswith('.txt'):
        image_paths, landmarks_list = loader.load_from_txt(annotations_file)
    else:
        raise ValueError("不支持的文件格式，请使用.json或.txt文件")
    
    # 验证数据
    image_paths, landmarks_list = loader.validate_data(image_paths, landmarks_list)
    
    if len(image_paths) == 0:
        raise ValueError("没有有效的训练数据！")
    
    # 显示数据统计
    stats = loader.get_statistics(landmarks_list)
    print(f"📊 数据统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  关键点范围: X[{stats['x_range'][0]:.1f}, {stats['x_range'][1]:.1f}], Y[{stats['y_range'][0]:.1f}, {stats['y_range'][1]:.1f}]")
    print(f"  关键点均值: X={stats['x_mean']:.1f}, Y={stats['y_mean']:.1f}")
    
    # 数据分割
    print("🔄 分割数据集...")
    
    # 首先分割出测试集
    if test_split > 0:
        train_val_paths, test_paths, train_val_landmarks, test_landmarks = train_test_split(
            image_paths, landmarks_list, test_size=test_split, random_state=42, stratify=None
        )
    else:
        train_val_paths, train_val_landmarks = image_paths, landmarks_list
        test_paths, test_landmarks = [], []
    
    # 然后从训练+验证集中分割出验证集
    if val_split > 0 and len(train_val_paths) > 0:
        train_paths, val_paths, train_landmarks, val_landmarks = train_test_split(
            train_val_paths, train_val_landmarks, test_size=val_split, random_state=42, stratify=None
        )
    else:
        train_paths, train_landmarks = train_val_paths, train_val_landmarks
        val_paths, val_landmarks = [], []
    
    print(f"  训练集: {len(train_paths)} 样本")
    print(f"  验证集: {len(val_paths)} 样本")
    print(f"  测试集: {len(test_paths)} 样本")
    
    # 创建数据变换
    train_transform, val_transform = create_data_transforms()
    
    # 创建数据集
    train_dataset = FishLandmarkDataset(train_paths, train_landmarks, train_transform)  # 使用所有关键点
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # 处理验证集
    if len(val_paths) > 0:
        val_dataset = FishLandmarkDataset(val_paths, val_landmarks, val_transform)  # 使用所有关键点
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        print("⚠️  验证集为空，将使用训练集的一部分作为验证集")
        # 从训练集中取一部分作为验证集
        val_size = min(len(train_paths) // 5, 50)  # 取训练集的1/5或最多50个样本
        if val_size > 0:
            val_paths = train_paths[:val_size]
            val_landmarks = train_landmarks[:val_size]
            train_paths = train_paths[val_size:]
            train_landmarks = train_landmarks[val_size:]
            
            # 重新创建数据集
            train_dataset = FishLandmarkDataset(train_paths, train_landmarks, train_transform)  # 使用所有关键点
            val_dataset = FishLandmarkDataset(val_paths, val_landmarks, val_transform)  # 使用所有关键点
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
            print(f"  重新分配 - 训练集: {len(train_paths)} 样本, 验证集: {len(val_paths)} 样本")
        else:
            val_loader = None
            print("⚠️  数据量太少，无法创建验证集")
    
    # 创建模型
    print("🏗️  创建模型...")
    detector = FishLandmarkDetector()
    model = detector.create_model(backbone=backbone)
    
    print(f"  模型架构: {backbone}")
    print(f"  关键点数量: {len(detector.landmark_names)}")
    print(f"  设备: {detector.device}")
    
    # 训练模型
    print("🚀 开始训练...")
    print(f"🔧 损失函数类型: {loss_type}")
    print(f"🔧 锐度参数: {sharpness} (值越大越锐利，惩罚越重)")
    print(f"🎯 预测关键点: {detector.landmark_names} (仅身体中心)")
    if val_loader is not None:
        train_losses, val_losses = detector.train_with_configurable_loss(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            save_dir=save_dir,
            sharpness=sharpness,
            loss_type=loss_type
        )
    else:
        print("⚠️  没有验证集，将只进行训练（不进行验证）")
        train_losses, val_losses = detector.train_without_validation_configurable_loss(
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            save_dir=save_dir,
            sharpness=sharpness,
            loss_type=loss_type
        )
    
    # 保存训练配置
    config = {
        'data_dir': data_dir,
        'annotations_file': annotations_file,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'backbone': backbone,
        'train_samples': len(train_paths),
        'val_samples': len(val_paths),
        'test_samples': len(test_paths),
        'landmark_names': detector.landmark_names,
        'loss_type': loss_type,
        'sharpness': sharpness,
        'training_date': datetime.now().isoformat()
    }
    
    config_path = os.path.join(save_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 训练完成！")
    print(f"📁 模型保存在: {save_dir}")
    print(f"📋 训练配置保存在: {config_path}")
    
    return detector, train_losses, val_losses


def test_model(model_path: str, test_data_dir: str, test_annotations: str, 
               output_dir: str = 'test_results', same_folder_mode: bool = False):
    """测试训练好的模型"""
    
    print("="*60)
    print("🧪 测试鱼体关键点检测模型")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    try:
        detector = FishLandmarkDetector(model_path=model_path)
        if detector.model is None:
            raise ValueError("模型加载失败，model为None")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 加载测试数据
    if same_folder_mode:
        print("📂 使用同文件夹模式：图像和JSON文件在同一目录")
    loader = FishLandmarkDataLoader(test_data_dir, same_folder_mode=same_folder_mode)
    if test_annotations.endswith('.json'):
        image_paths, landmarks_list = loader.load_from_json(test_annotations)
    else:
        image_paths, landmarks_list = loader.load_from_txt(test_annotations)
    
    image_paths, landmarks_list = loader.validate_data(image_paths, landmarks_list)
    
    print(f"📊 测试样本数: {len(image_paths)}")
    
    if len(image_paths) == 0:
        print("❌ 没有有效的测试数据")
        return
    
    # 测试模型
    detector.model.eval()
    errors = []
    
    # 创建与训练相同的数据变换
    _, val_transform = create_data_transforms((256, 256))
    
    # 创建测试数据集（使用与训练相同的预处理）
    test_dataset = FishLandmarkDataset(image_paths, landmarks_list, val_transform)  # 使用所有关键点
    
    for i in range(len(test_dataset)):
        # 从数据集获取样本（已经过预处理）
        sample = test_dataset[i]
        image_tensor = sample['image']
        true_landmarks_normalized = sample['landmarks'].numpy()
        image_path = sample['image_path']
        original_size = sample['original_size']
        
        # 加载原始图像用于预测（predict方法会自己处理预处理）
        original_image = cv2.imread(image_path)
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # 预测关键点（predict方法会处理预处理和坐标转换）
        pred_landmarks, pred_visibility = detector.predict(original_image_rgb)
        
        # 将真实关键点从归一化坐标转换回像素坐标
        # 使用与predict方法相同的转换逻辑
        original_h, original_w = original_size
        target_size = (256, 256)
        scale = min(target_size[0] / original_w, target_size[1] / original_h)
        true_landmarks_pixel = true_landmarks_normalized * np.array([target_size[0], target_size[1]]) / scale
        
        # 调试信息（前几个样本）
        if i < 3:
            print(f"  调试样本 {i+1}:")
            print(f"    原始图像尺寸: {original_image_rgb.shape}")
            print(f"    原始尺寸: {original_size}")
            print(f"    缩放比例: {scale:.3f}")
            print(f"    真实关键点(归一化): {true_landmarks_normalized}")
            print(f"    真实关键点(像素): {true_landmarks_pixel}")
            print(f"    预测关键点(像素): {pred_landmarks}")
            print(f"    可见性: {pred_visibility}")
        
        # 计算误差（现在都在像素坐标系中）
        error = np.linalg.norm(pred_landmarks - true_landmarks_pixel, axis=1)
        errors.append(error)
        
        # 可视化结果
        vis_image = detector.visualize_landmarks(original_image_rgb, pred_landmarks, pred_visibility)
        
        # 保存结果
        image_name = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"result_{i:03d}_{image_name}")
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # 计算鱼的精确中心（使用身体中心）
        fish_center = detector.calculate_fish_center(pred_landmarks, pred_visibility)
        true_center = detector.calculate_fish_center(true_landmarks_pixel, np.ones_like(pred_visibility))
        
        # 显示身体中心和头部中心的预测结果
        pred_body = pred_landmarks[0] if len(pred_landmarks) > 0 else [0, 0]
        pred_head = pred_landmarks[1] if len(pred_landmarks) > 1 else [0, 0]
        true_body = true_landmarks_pixel[0] if len(true_landmarks_pixel) > 0 else [0, 0]
        true_head = true_landmarks_pixel[1] if len(true_landmarks_pixel) > 1 else [0, 0]
        
        print(f"样本 {i+1:3d}: 身体中心 预测=({pred_body[0]:.1f}, {pred_body[1]:.1f}) 真实=({true_body[0]:.1f}, {true_body[1]:.1f}) "
              f"头部中心 预测=({pred_head[0]:.1f}, {pred_head[1]:.1f}) 真实=({true_head[0]:.1f}, {true_head[1]:.1f}) "
              f"中心误差={np.linalg.norm(fish_center - true_center):.1f}px")
    
    # 计算总体统计
    all_errors = np.concatenate(errors)
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    max_error = np.max(all_errors)
    
    print(f"\n📈 测试结果统计:")
    print(f"  平均误差: {mean_error:.2f} ± {std_error:.2f} 像素")
    print(f"  最大误差: {max_error:.2f} 像素")
    print(f"  结果图像保存在: {output_dir}")
    
    return mean_error, std_error, max_error


def print_usage_examples():
    """打印详细的使用示例"""
    print("""
📖 鱼体关键点检测模型训练脚本使用示例:

🚀 训练模式示例:

1. 基本训练:
   python3 train_landmark_model.py --mode train \
       --data_dir ./process_data \
       --annotations ./process_data/train_annotations.json \
       --epochs 100

2. 自定义参数训练:
   python3 train_landmark_model.py --mode train \
       --data_dir ./process_data \
       --annotations ./process_data/train_annotations.json \
       --epochs 100 \
       --batch_size 128 \
       --lr 0.0005 \
       --backbone efficientnet \
       --exp_name gaussian_$(date +%Y%m%d_%H%M%S)

3. 指定实验名称 (自动生成时间戳目录):
   python3 train_landmark_model.py --mode train \
       --data_dir ./process_data \
       --annotations ./process_data/train_annotations.json \
       --exp_name body_center_only \
       --epochs 50

4. 快速训练 (少量epochs):
   python train_landmark_model.py --mode train \
       --data_dir ./landmarks/processed_data \
       --annotations train_annotations.json \
       --epochs 50 \
       --batch_size 8

5. 高锐度椭圆核训练 (更严格的惩罚):
   python3 train_landmark_model.py --mode train \
       --data_dir ./process_data \
       --annotations ./process_data/train_annotations.json \
       --epochs 100 \
       --sharpness 3.0 \
       --loss_type ellipsoid

6. 高斯核损失训练:
   python3 train_landmark_model.py --mode train \
       --data_dir ./process_data \
       --annotations ./process_data/train_annotations.json \
       --epochs 100 \
       --loss_type gaussian \
       --sharpness 2.0

🧪 测试模式示例:

1. 基本测试:
   python3 train_landmark_model.py --mode test \
       --model_path ./models/best_fish_landmark_model.pth \
       --test_data_dir ./process_data \
       --test_annotations ./process_data/val_annotations.json

2. 自定义输出目录测试:
   python3 train_landmark_model.py --mode test \
       --model_path ./experiments/gaussian_20250922_153626_20250922_153630/best_fish_landmark_model_gaussian.pth \
       --test_data_dir ./process_data \
       --test_annotations ./process_data/train_annotations.json \
       --output_dir ./my_test_results

    python3 train_landmark_model.py --mode test \
       --model_path ./experiments/ellipsoid_20250922_130057_20250922_130101/best_fish_landmark_model_ellipsoid.pth \
       --test_data_dir ./process_data \
       --test_annotations ./process_data/val_annotations.json \
       --output_dir ./my_test_results_ellipsoid

📁 数据目录结构要求:
   data_dir/
   ├── images/                    # 图像文件
   │   ├── train_image1.jpg
   │   ├── train_image2.jpg
   │   └── ...
   └── landmarks/                 # 关键点numpy文件
       ├── train_image1.npy
       ├── train_image2.npy
       └── ...

📄 标注文件格式:
   # JSON格式 (推荐) - 仅使用身体中心
   {
     "train_image1.jpg": {
       "landmarks": [[100, 50], [100, 150]],  # [头部中心, 身体中心] - 仅使用身体中心
       "visibility": [1, 1]
     },
     "train_image2.jpg": {
       "landmarks": [[120, 60], [120, 160]],  # [头部中心, 身体中心] - 仅使用身体中心
       "visibility": [1, 1]
     }
   }

🔧 参数说明:
   --mode: 运行模式 (train/test)
   --data_dir: 数据目录路径
   --annotations: 标注文件路径
   --epochs: 训练轮数 (默认: 100)
   --batch_size: 批次大小 (默认: 16)
   --lr: 学习率 (默认: 0.001)
   --backbone: 模型架构 (resnet18/efficientnet, 默认: resnet18)
   --save_dir: 模型保存目录 (默认: models)
   --test_split: 测试集比例 (默认: 0.2)
   --val_split: 验证集比例 (默认: 0.2)

⚠️  注意事项:
   - 确保数据目录包含images和landmarks子目录
   - 图像文件名和标注文件中的键名必须匹配
   - 关键点坐标格式: [[x1, y1], [x2, y2]] (头部中心, 身体中心) - 仅使用身体中心
   - 可见性格式: [1, 1] (1=可见, 0=不可见)
   - 训练过程中会自动保存最佳模型和训练配置
""")


def main():
    parser = argparse.ArgumentParser(
        description='鱼体关键点检测模型训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --mode train --data_dir ./data --annotations train_annotations.json --epochs 100
  %(prog)s --mode test --model_path ./models/best_fish_landmark_model.pth --test_data_dir ./test_data --test_annotations test_annotations.json
  %(prog)s --help  # 查看详细使用说明
        """
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, 
                       help='运行模式: train(训练) 或 test(测试)')
    
    # 训练参数
    parser.add_argument('--data_dir', type=str, 
                       help='数据目录 (包含images和landmarks子目录)')
    parser.add_argument('--annotations', type=str, 
                       help='标注文件路径 (JSON或TXT格式)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='批次大小 (默认: 16)')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                       choices=['resnet18', 'efficientnet'], 
                       help='模型架构 (默认: resnet18)')
    parser.add_argument('--save_dir', type=str, default=None, 
                       help='模型保存目录 (默认: 自动生成带时间戳的实验目录)')
    parser.add_argument('--exp_name', type=str, default='fish_landmark', 
                       help='实验名称 (默认: fish_landmark)')
    parser.add_argument('--test_split', type=float, default=0.2, 
                       help='测试集比例 (默认: 0.2)')
    parser.add_argument('--val_split', type=float, default=0.2, 
                       help='验证集比例 (默认: 0.2)')
    parser.add_argument('--same_folder_mode', action='store_true', 
                       help='同文件夹模式：图像和JSON文件在同一目录 (默认: False)')
    parser.add_argument('--sharpness', type=float, default=1.0, 
                       help='锐度参数 (默认: 1.0, 值越大越锐利，惩罚越重)')
    parser.add_argument('--loss_type', type=str, default='ellipsoid', 
                       choices=['gaussian', 'ellipsoid'],
                       help='损失函数类型 (默认: ellipsoid) - gaussian: 高斯核损失, ellipsoid: 椭圆核损失')
    
    # 测试参数
    parser.add_argument('--model_path', type=str, 
                       help='模型文件路径 (.pth文件)')
    parser.add_argument('--test_data_dir', type=str, 
                       help='测试数据目录')
    parser.add_argument('--test_annotations', type=str, 
                       help='测试标注文件')
    parser.add_argument('--output_dir', type=str, default='test_results', 
                       help='测试结果输出目录 (默认: test_results)')
    
    args = parser.parse_args()
    
    # 生成实验目录名称
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"experiments/{args.exp_name}_{timestamp}"
    
    print(f"📁 实验目录: {args.save_dir}")
    
    # 检查是否是帮助请求
    if '--help' in sys.argv or '-h' in sys.argv:
        print_usage_examples()
        return
    
    if args.mode == 'train':
        if not args.data_dir or not args.annotations:
            print("❌ 错误: 训练模式需要指定 --data_dir 和 --annotations")
            print("\n💡 使用示例:")
            print("python train_landmark_model.py --mode train --data_dir ./data --annotations train_annotations.json")
            print("\n📖 查看详细帮助:")
            print("python train_landmark_model.py --help")
            return
        
        print("🚀 开始训练模式...")
        train_model(
            data_dir=args.data_dir,
            annotations_file=args.annotations,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            backbone=args.backbone,
            save_dir=args.save_dir,
            test_split=args.test_split,
            val_split=args.val_split,
            sharpness=args.sharpness,
            loss_type=args.loss_type
        )
    
    elif args.mode == 'test':
        if not args.model_path or not args.test_data_dir or not args.test_annotations:
            print("❌ 错误: 测试模式需要指定 --model_path, --test_data_dir 和 --test_annotations")
            print("\n💡 使用示例:")
            print("python train_landmark_model.py --mode test --model_path ./models/best_model.pth --test_data_dir ./test_data --test_annotations test_annotations.json")
            print("\n📖 查看详细帮助:")
            print("python train_landmark_model.py --help")
            return
        
        print("🧪 开始测试模式...")
        test_model(
            model_path=args.model_path,
            test_data_dir=args.test_data_dir,
            test_annotations=args.test_annotations,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    import sys
    
    # 如果没有参数，显示使用示例
    if len(sys.argv) == 1:
        print_usage_examples()
        sys.exit(0)
    
    main()
