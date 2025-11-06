#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 YOLO 数据集的 images 和 labels 按比例分配到 train/test/val 子文件夹
确保 images 和 labels 中对应文件名称一致
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple


def get_image_base_name(image_path: Path) -> str:
    """获取图片的基础名称（不含扩展名）"""
    return image_path.stem


def find_matching_label(image_base_name: str, labels_dir: Path) -> Path:
    """根据图片名称找到对应的标签文件"""
    label_path = labels_dir / f"{image_base_name}.txt"
    if label_path.exists():
        return label_path
    return None


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_base: Path,
    train_ratio: float = 0.7,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    将数据集按比例分配到 train/test/val
    
    Args:
        images_dir: 源图片目录
        labels_dir: 源标签目录
        output_base: 输出基础目录（将在此创建 images 和 labels 子目录）
        train_ratio: 训练集比例
        test_ratio: 测试集比例
        val_ratio: 验证集比例
        seed: 随机种子
    """
    # 验证比例
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6, "比例总和必须为1.0"
    
    # 设置随机种子
    random.seed(seed)
    
    # 收集所有图片文件
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = []
    for ext in image_exts:
        image_files.extend(list(images_dir.rglob(f"*{ext}")))
        image_files.extend(list(images_dir.rglob(f"*{ext.upper()}")))
    
    # 过滤：只保留有对应标签文件的图片
    valid_pairs: List[Tuple[Path, Path]] = []
    for img_path in image_files:
        base_name = get_image_base_name(img_path)
        label_path = find_matching_label(base_name, labels_dir)
        if label_path and label_path.exists():
            valid_pairs.append((img_path, label_path))
        else:
            print(f"[警告] 未找到图片 {img_path.name} 对应的标签文件，跳过")
    
    print(f"找到 {len(valid_pairs)} 个有效的图片-标签对")
    
    # 随机打乱
    random.shuffle(valid_pairs)
    
    # 计算分割点
    total = len(valid_pairs)
    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    val_count = total - train_count - test_count  # 剩余的全部给val
    
    print(f"分配方案: train={train_count} ({train_count/total*100:.1f}%), "
          f"test={test_count} ({test_count/total*100:.1f}%), "
          f"val={val_count} ({val_count/total*100:.1f}%)")
    
    # 分割
    train_pairs = valid_pairs[:train_count]
    test_pairs = valid_pairs[train_count:train_count + test_count]
    val_pairs = valid_pairs[train_count + test_count:]
    
    # 创建输出目录结构
    splits = {
        "train": train_pairs,
        "test": test_pairs,
        "val": val_pairs
    }
    
    for split_name, pairs in splits.items():
        images_out = output_base / "images" / split_name
        labels_out = output_base / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        
        for img_path, label_path in pairs:
            # 复制图片
            img_dst = images_out / img_path.name
            shutil.copy2(img_path, img_dst)
            
            # 复制标签
            label_dst = labels_out / label_path.name
            shutil.copy2(label_path, label_dst)
        
        print(f"[完成] {split_name}: {len(pairs)} 个文件")
    
    print(f"\n数据集分割完成！")
    print(f"输出目录: {output_base}")
    print(f"结构:")
    print(f"  {output_base}/")
    print(f"    images/")
    print(f"      train/ ({len(train_pairs)} 个文件)")
    print(f"      test/ ({len(test_pairs)} 个文件)")
    print(f"      val/ ({len(val_pairs)} 个文件)")
    print(f"    labels/")
    print(f"      train/ ({len(train_pairs)} 个文件)")
    print(f"      test/ ({len(test_pairs)} 个文件)")
    print(f"      val/ ({len(val_pairs)} 个文件)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="将YOLO数据集按比例分配到train/test/val")
    parser.add_argument("--images_dir", required=True, help="源图片目录")
    parser.add_argument("--labels_dir", required=True, help="源标签目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例 (默认: 0.7)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="测试集比例 (默认: 0.15)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例 (默认: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    
    if not images_dir.exists():
        print(f"[错误] 图片目录不存在: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"[错误] 标签目录不存在: {labels_dir}")
        return
    
    split_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_base=output_dir,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()


