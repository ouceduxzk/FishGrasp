#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分割脚本 - 将同文件夹的图像和标注数据分割为训练集和验证集

使用示例:
    # 基本用法
    python3 split_data.py --input_dir ./raw_data --output_dir ./processed_data --val_ratio 0.2
    
    # 自定义参数
    python3 split_data.py --input_dir ./my_data --output_dir ./split_data --val_ratio 0.3 --random_seed 42
    
    # 查看帮助
    python3 split_data.py --help

功能特性:
    - 支持图像和JSON标注文件在同一目录的结构
    - 自动创建训练集和验证集目录结构
    - 支持自定义验证集比例
    - 支持随机种子设置以确保可重现性
    - 自动处理图像和标注文件的配对
    - 生成分割统计信息
"""

import os
import json
import shutil
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np


def find_image_annotation_pairs(input_dir: Path) -> List[Tuple[Path, Path]]:
    """
    查找图像和标注文件的配对
    
    Args:
        input_dir: 输入目录路径
        
    Returns:
        图像和标注文件路径的配对列表
    """
    pairs = []
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    annotation_extensions = {'.json'}
    
    # 获取所有文件
    all_files = list(input_dir.iterdir())
    
    # 分离图像文件和标注文件
    image_files = {}
    annotation_files = {}
    
    for file_path in all_files:
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in image_extensions:
                # 使用文件名（不含扩展名）作为键
                key = file_path.stem
                image_files[key] = file_path
            elif ext in annotation_extensions:
                key = file_path.stem
                annotation_files[key] = file_path
    
    # 配对图像和标注文件
    for key in image_files:
        if key in annotation_files:
            pairs.append((image_files[key], annotation_files[key]))
        else:
            print(f"⚠️  警告: 图像文件 {image_files[key].name} 没有对应的标注文件")
    
    # 检查未配对的标注文件
    for key in annotation_files:
        if key not in image_files:
            print(f"⚠️  警告: 标注文件 {annotation_files[key].name} 没有对应的图像文件")
    
    return pairs


def validate_annotation_file(annotation_path: Path) -> bool:
    """
    验证标注文件格式是否正确
    
    Args:
        annotation_path: 标注文件路径
        
    Returns:
        是否为有效的标注文件
    """
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否为字典格式
        if not isinstance(data, dict):
            print(f"❌ 标注文件格式错误: {annotation_path.name} (应为字典格式)")
            return False
        
        # 检查每个条目的格式
        for image_name, annotation in data.items():
            if not isinstance(annotation, dict):
                print(f"❌ 标注文件格式错误: {annotation_path.name} (条目应为字典)")
                return False
            
            if 'landmarks' not in annotation:
                print(f"❌ 标注文件格式错误: {annotation_path.name} (缺少landmarks字段)")
                return False
            
            landmarks = annotation['landmarks']
            if not isinstance(landmarks, list) or len(landmarks) == 0:
                print(f"❌ 标注文件格式错误: {annotation_path.name} (landmarks应为非空列表)")
                return False
            
            # 检查关键点格式
            for landmark in landmarks:
                if not isinstance(landmark, list) or len(landmark) != 2:
                    print(f"❌ 标注文件格式错误: {annotation_path.name} (关键点应为[x,y]格式)")
                    return False
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {annotation_path.name} - {e}")
        return False
    except Exception as e:
        print(f"❌ 文件读取错误: {annotation_path.name} - {e}")
        return False


def split_data(input_dir: str, output_dir: str, val_ratio: float = 0.2, 
               random_seed: int = 42) -> None:
    """
    分割数据为训练集和验证集
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        val_ratio: 验证集比例
        random_seed: 随机种子
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print("="*60)
    print("📂 数据分割工具")
    print("="*60)
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"验证集比例: {val_ratio:.1%}")
    print(f"随机种子: {random_seed}")
    
    # 检查输入目录
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_path}")
    
    if not input_path.is_dir():
        raise ValueError(f"输入路径不是目录: {input_path}")
    
    # 查找图像和标注文件配对
    print("\n🔍 查找图像和标注文件配对...")
    pairs = find_image_annotation_pairs(input_path)
    
    if len(pairs) == 0:
        raise ValueError("未找到有效的图像-标注文件配对")
    
    print(f"✅ 找到 {len(pairs)} 个有效的图像-标注文件配对")
    
    # 验证标注文件
    print("\n🔍 验证标注文件格式...")
    valid_pairs = []
    for image_path, annotation_path in pairs:
        if validate_annotation_file(annotation_path):
            valid_pairs.append((image_path, annotation_path))
        else:
            print(f"❌ 跳过无效的标注文件: {annotation_path.name}")
    
    if len(valid_pairs) == 0:
        raise ValueError("没有有效的标注文件")
    
    print(f"✅ {len(valid_pairs)} 个标注文件格式正确")
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 随机打乱数据
    random.shuffle(valid_pairs)
    
    # 计算分割点
    total_count = len(valid_pairs)
    val_count = int(total_count * val_ratio)
    train_count = total_count - val_count
    
    print(f"\n📊 数据分割统计:")
    print(f"  总样本数: {total_count}")
    print(f"  训练集: {train_count} 样本 ({train_count/total_count:.1%})")
    print(f"  验证集: {val_count} 样本 ({val_count/total_count:.1%})")
    
    # 分割数据
    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:]
    
    # 创建输出目录结构
    print(f"\n📁 创建输出目录结构...")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    train_images_dir = output_path / "images" / "train"
    val_images_dir = output_path / "images" / "val"
    train_labels_dir = output_path / "labels" / "train"
    val_labels_dir = output_path / "labels" / "val"
    
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 复制训练集文件
    print(f"\n📋 复制训练集文件...")
    train_annotations = {}
    
    for i, (image_path, annotation_path) in enumerate(train_pairs):
        # 复制图像文件
        train_image_path = train_images_dir / image_path.name
        shutil.copy2(image_path, train_image_path)
        
        # 复制标注文件
        train_label_path = train_labels_dir / annotation_path.name
        shutil.copy2(annotation_path, train_label_path)
        
        # 收集训练集标注信息
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        train_annotations.update(annotation_data)
        
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(train_pairs)} 个训练样本")
    
    # 复制验证集文件
    print(f"\n📋 复制验证集文件...")
    val_annotations = {}
    
    for i, (image_path, annotation_path) in enumerate(val_pairs):
        # 复制图像文件
        val_image_path = val_images_dir / image_path.name
        shutil.copy2(image_path, val_image_path)
        
        # 复制标注文件
        val_label_path = val_labels_dir / annotation_path.name
        shutil.copy2(annotation_path, val_label_path)
        
        # 收集验证集标注信息
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        val_annotations.update(annotation_data)
        
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(val_pairs)} 个验证样本")
    
    # 生成合并的标注文件
    print(f"\n📄 生成合并的标注文件...")
    
    # 训练集标注文件
    train_annotations_path = output_path / "train_annotations.json"
    with open(train_annotations_path, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2, ensure_ascii=False)
    
    # 验证集标注文件
    val_annotations_path = output_path / "val_annotations.json"
    with open(val_annotations_path, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=2, ensure_ascii=False)
    
    # 生成分割信息文件
    split_info = {
        "total_samples": total_count,
        "train_samples": train_count,
        "val_samples": val_count,
        "val_ratio": val_ratio,
        "random_seed": random_seed,
        "input_dir": str(input_path),
        "output_dir": str(output_path)
    }
    
    split_info_path = output_path / "split_info.json"
    with open(split_info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 数据分割完成!")
    print(f"📁 输出目录结构:")
    print(f"  {output_path}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/ ({train_count} 个图像)")
    print(f"  │   └── val/ ({val_count} 个图像)")
    print(f"  ├── labels/")
    print(f"  │   ├── train/ ({train_count} 个标注)")
    print(f"  │   └── val/ ({val_count} 个标注)")
    print(f"  ├── train_annotations.json")
    print(f"  ├── val_annotations.json")
    print(f"  └── split_info.json")
    
    print(f"\n🚀 现在可以使用以下命令进行训练:")
    print(f"python3 train_landmark_model.py --mode train \\")
    print(f"    --data_dir {output_path} \\")
    print(f"    --annotations {train_annotations_path} \\")
    print(f"    --epochs 100")


def main():
    parser = argparse.ArgumentParser(
        description='数据分割工具 - 将同文件夹的图像和标注数据分割为训练集和验证集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --input_dir ./raw_data --output_dir ./processed_data --val_ratio 0.2
  %(prog)s --input_dir ./my_data --output_dir ./split_data --val_ratio 0.3 --random_seed 42
  %(prog)s --help  # 查看详细帮助

输入目录结构:
  input_dir/
  ├── image1.jpg
  ├── image1.json
  ├── image2.jpg
  ├── image2.json
  └── ...

输出目录结构:
  output_dir/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/
  │   └── val/
  ├── train_annotations.json
  ├── val_annotations.json
  └── split_info.json
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入目录路径 (包含图像和JSON标注文件)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径 (将创建训练集和验证集目录结构)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例 (默认: 0.2)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 验证参数
    if not 0 < args.val_ratio < 1:
        print("❌ 错误: val_ratio 必须在 0 和 1 之间")
        return
    
    try:
        split_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            random_seed=args.random_seed
        )
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
