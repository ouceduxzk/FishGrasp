#!/usr/bin/env python3
"""
简单的关键点预测脚本 - 在单张图像上预测鱼的身体中心

使用方法:
    python3 predict_landmarks.py --model_path ./models/best_model.pth --image_path ./test_image.jpg
    python3 predict_landmarks.py --model_path ./models/best_model.pth --image_path ./test_image.jpg --output_dir ./results
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 导入关键点检测器
from fish_landmark_detector import FishLandmarkDetector

def predict_single_image(model_path, image_path, output_dir=None, device='auto'):
    """
    在单张图像上预测关键点
    
    Args:
        model_path: 模型文件路径 (.pth)
        image_path: 输入图像路径
        output_dir: 输出目录（可选，用于保存结果）
        device: 设备 ('auto', 'cpu', 'cuda')
    """
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return False
    
    # 自动选择设备
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔧 使用设备: {device}")
    print(f"📁 模型路径: {model_path}")
    print(f"🖼️  图像路径: {image_path}")
    
    try:
        # 初始化检测器
        print("🚀 正在加载模型...")
        detector = FishLandmarkDetector(model_path=model_path, device=device)
        print("✅ 模型加载成功")
        
        # 读取图像
        print("📖 正在读取图像...")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"❌ 无法读取图像: {image_path}")
            return False
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        print(f"📐 图像尺寸: {image_rgb.shape}")
        
        # 预测关键点
        print("🔍 正在预测关键点...")
        landmarks, visibility = detector.predict(image_rgb)
        
        # 计算鱼的中心点
        center = detector.calculate_fish_center(landmarks, visibility)
        
        print("🎯 预测结果:")
        print(f"  关键点坐标: {landmarks}")
        print(f"  可见性: {visibility}")
        print(f"  鱼中心点: {center}")
        
        # 可视化结果
        vis_image = detector.visualize_landmarks(image_rgb, landmarks, visibility)
        
        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存可视化图像
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(output_dir, f"prediction_{Path(image_path).stem}.jpg")
            cv2.imwrite(output_path, vis_bgr)
            print(f"💾 结果已保存到: {output_path}")
            
            # 保存预测数据
            result_data = {
                'image_path': image_path,
                'model_path': model_path,
                'landmarks': landmarks.tolist() if hasattr(landmarks, 'tolist') else landmarks,
                'visibility': visibility.tolist() if hasattr(visibility, 'tolist') else visibility,
                'fish_center': center.tolist() if hasattr(center, 'tolist') else center
            }
            
            import json
            json_path = os.path.join(output_dir, f"prediction_{Path(image_path).stem}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            print(f"📊 预测数据已保存到: {json_path}")
        
        # 显示结果（如果可能）
        try:
            cv2.imshow('Landmark Prediction', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print("👀 按任意键关闭预览窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"⚠️  无法显示预览窗口: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='在单张图像上预测鱼的关键点',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本预测
  python3 predict_landmarks.py --model_path ./models/best_model.pth --image_path ./test.jpg
  
  # 保存结果到指定目录
  python3 predict_landmarks.py --model_path ./models/best_model.pth --image_path ./test.jpg --output_dir ./results
  
  # 指定设备
  python3 predict_landmarks.py --model_path ./models/best_model.pth --image_path ./test.jpg --device cuda
  
  # 批量预测（使用通配符）
  python3 predict_landmarks.py --model_path ./models/best_model.pth --image_path "./images/*.jpg" --output_dir ./results
        """
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型文件路径 (.pth)')
    parser.add_argument('--image_path', type=str, required=True,
                        help='输入图像路径（支持通配符，如 "./images/*.jpg"）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（可选，用于保存预测结果）')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='运行设备 (默认: auto)')
    
    args = parser.parse_args()
    
    # 处理通配符路径
    import glob
    image_paths = glob.glob(args.image_path)
    
    if not image_paths:
        print(f"❌ 未找到匹配的图像文件: {args.image_path}")
        return
    
    print(f"🔍 找到 {len(image_paths)} 张图像")
    
    # 预测每张图像
    success_count = 0
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n{'='*60}")
        print(f"📸 处理图像 {i}/{len(image_paths)}: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        if predict_single_image(args.model_path, image_path, args.output_dir, args.device):
            success_count += 1
    
    print(f"\n🎉 处理完成！成功预测 {success_count}/{len(image_paths)} 张图像")

if __name__ == "__main__":
    main()
