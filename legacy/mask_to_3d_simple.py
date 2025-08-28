#!/usr/bin/env python3
"""
从掩码文件夹加载掩码，提取鱿鱼对象，使用RealSense相机内参转换为3D点云并保存。
简化版本，不依赖 open3d，直接保存为 PLY 格式。

使用方法:
    python mask_to_3d_simple.py --mask_dir arbitray_view/mask --rgb_dir arbitray_view/rgb --depth_dir arbitray_view/depth --output_dir pointclouds_new

依赖:
    - numpy, opencv-python, pillow
"""

import argparse
import os
import sys
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import json

def load_camera_intrinsics(intrinsics_file=None):
    """
    加载相机内参
    
    Args:
        intrinsics_file: 内参文件路径，如果为None则使用默认的RealSense D435i参数
    
    Returns:
        fx, fy, cx, cy: 相机内参
    """
    if intrinsics_file and os.path.exists(intrinsics_file):
        with open(intrinsics_file, 'r') as f:
            intrinsics = json.load(f)
            fx = intrinsics['fx']
            fy = intrinsics['fy']
            cx = intrinsics['cx']
            cy = intrinsics['cy']
    else:
        # 默认RealSense D435i参数 (640x480)
        fx = 615.0  # 焦距x
        fy = 615.0  # 焦距y
        cx = 320.0  # 主点x
        cy = 240.0  # 主点y
    
    return fx, fy, cx, cy

def load_depth_image(depth_path):
    """
    加载深度图像
    
    Args:
        depth_path: 深度图像路径
    
    Returns:
        depth_array: 深度数组 (单位: 米)
    """
    # 尝试不同的深度图像格式
    if depth_path.endswith('.png'):
        # 16位PNG深度图像
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_img is None:
            raise FileNotFoundError(f"无法读取深度图像: {depth_path}")
        
        # 转换为米为单位 (假设深度图像单位为毫米)
        depth_array = depth_img.astype(np.float32) / 1000.0
    else:
        # 其他格式
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            raise FileNotFoundError(f"无法读取深度图像: {depth_path}")
        depth_array = depth_img.astype(np.float32) / 1000.0
    
    return depth_array

def mask_to_3d_pointcloud(rgb_image, depth_image, mask, fx, fy, cx, cy, min_depth=0.1, max_depth=10.0):
    """
    将掩码区域转换为3D点云
    
    Args:
        rgb_image: RGB图像 (H, W, 3)
        depth_image: 深度图像 (H, W) 单位: 米
        mask: 二值掩码 (H, W)
        fx, fy, cx, cy: 相机内参
        min_depth: 最小深度阈值 (米)
        max_depth: 最大深度阈值 (米)
    
    Returns:
        points: 3D点坐标 (N, 3)
        colors: RGB颜色 (N, 3)
    """
    height, width = depth_image.shape
    
    # 创建网格坐标
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # 应用掩码
    mask_indices = np.where(mask > 0)
    x_coords = x_coords[mask_indices]
    y_coords = y_coords[mask_indices]
    depths = depth_image[mask_indices]
    
    # 过滤无效深度
    valid_depth = (depths > min_depth) & (depths < max_depth)
    x_coords = x_coords[valid_depth]
    y_coords = y_coords[valid_depth]
    depths = depths[valid_depth]
    
    if len(depths) == 0:
        return np.array([]), np.array([])
    
    # 计算3D坐标
    z = depths
    x = (x_coords - cx) * z / fx
    y = (y_coords - cy) * z / fy
    
    # 组合3D点
    points = np.column_stack([x, y, z])
    
    # 获取对应的RGB颜色
    colors = rgb_image[mask_indices][valid_depth]
    colors = colors.astype(np.float32) / 255.0  # 归一化到[0,1]
    
    return points, colors

def save_pointcloud_ply(points, colors, output_path):
    """
    保存点云为PLY文件 (不依赖 open3d)
    
    Args:
        points: 3D点坐标 (N, 3)
        colors: RGB颜色 (N, 3)
        output_path: 输出文件路径
    """
    if len(points) == 0:
        print(f"警告: 没有有效的3D点，跳过保存: {output_path}")
        return
    
    # 将颜色从[0,1]转换为[0,255]
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    # 写入PLY文件
    with open(output_path, 'w') as f:
        # 写入PLY头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入点云数据
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors_uint8[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"已保存点云: {output_path} (点数: {len(points)})")

def process_single_image(mask_path, rgb_path, depth_path, output_path, fx, fy, cx, cy):
    """
    处理单张图像
    
    Args:
        mask_path: 掩码文件路径
        rgb_path: RGB图像路径
        depth_path: 深度图像路径
        output_path: 输出点云路径
        fx, fy, cx, cy: 相机内参
    """
    try:
        # 加载掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"错误: 无法读取掩码: {mask_path}")
            return False
        
        # 二值化掩码
        mask = (mask > 127).astype(np.uint8)
        
        # 加载RGB图像
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_image is None:
            print(f"错误: 无法读取RGB图像: {rgb_path}")
            return False
        
        # 转换为RGB格式
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # 加载深度图像
        depth_image = load_depth_image(depth_path)
        
        # 检查图像尺寸是否匹配
        if rgb_image.shape[:2] != depth_image.shape or rgb_image.shape[:2] != mask.shape:
            print(f"错误: 图像尺寸不匹配 - RGB: {rgb_image.shape}, Depth: {depth_image.shape}, Mask: {mask.shape}")
            return False
        
        # 转换为3D点云
        points, colors = mask_to_3d_pointcloud(
            rgb_image, depth_image, mask, fx, fy, cx, cy
        )
        
        # 保存点云
        save_pointcloud_ply(points, colors, output_path)
        
        return True
        
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return False

def find_matching_files(mask_dir, rgb_dir, depth_dir):
    """
    找到匹配的掩码、RGB和深度文件
    
    Args:
        mask_dir: 掩码目录
        rgb_dir: RGB图像目录
        depth_dir: 深度图像目录
    
    Returns:
        file_pairs: 匹配的文件对列表
    """
    # 获取所有掩码文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_mask.png')]
    
    file_pairs = []
    for mask_file in mask_files:
        # 提取基础文件名
        base_name = mask_file.replace('_mask.png', '')
        
        # 构建对应的RGB和深度文件路径
        rgb_file = f"{base_name}.png"
        depth_file = f"depth_{base_name.replace('rgb_', '')}.png"
        
        rgb_path = os.path.join(rgb_dir, rgb_file)
        depth_path = os.path.join(depth_dir, depth_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        # 检查文件是否存在
        if os.path.exists(rgb_path) and os.path.exists(depth_path):
            file_pairs.append({
                'mask': mask_path,
                'rgb': rgb_path,
                'depth': depth_path,
                'base_name': base_name
            })
        else:
            print(f"警告: 找不到匹配的文件 - RGB: {rgb_path}, Depth: {depth_path}")
    
    return file_pairs

def main():
    parser = argparse.ArgumentParser(description='从掩码生成3D点云 (简化版本)')
    parser.add_argument('--mask_dir', type=str, required=True,
                      help='掩码文件夹路径')
    parser.add_argument('--rgb_dir', type=str, required=True,
                      help='RGB图像文件夹路径')
    parser.add_argument('--depth_dir', type=str, required=True,
                      help='深度图像文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出点云文件夹路径')
    parser.add_argument('--intrinsics_file', type=str, default=None,
                      help='相机内参JSON文件路径')
    parser.add_argument('--min_depth', type=float, default=0.1,
                      help='最小深度阈值 (米)')
    parser.add_argument('--max_depth', type=float, default=10.0,
                      help='最大深度阈值 (米)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.mask_dir):
        print(f"错误: 掩码目录不存在: {args.mask_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.rgb_dir):
        print(f"错误: RGB目录不存在: {args.rgb_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.depth_dir):
        print(f"错误: 深度目录不存在: {args.depth_dir}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载相机内参
    fx, fy, cx, cy = load_camera_intrinsics(args.intrinsics_file)
    print(f"使用相机内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # 找到匹配的文件
    file_pairs = find_matching_files(args.mask_dir, args.rgb_dir, args.depth_dir)
    
    if not file_pairs:
        print("错误: 没有找到匹配的文件对")
        sys.exit(1)
    
    print(f"找到 {len(file_pairs)} 个匹配的文件对")
    
    # 处理每个文件对
    success_count = 0
    for file_pair in tqdm(file_pairs, desc="处理图像"):
        output_path = os.path.join(args.output_dir, f"{file_pair['base_name']}_pointcloud.ply")
        
        success = process_single_image(
            file_pair['mask'],
            file_pair['rgb'],
            file_pair['depth'],
            output_path,
            fx, fy, cx, cy
        )
        
        if success:
            success_count += 1
    
    print(f"\n处理完成! 成功处理 {success_count}/{len(file_pairs)} 个文件")
    print(f"点云文件保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
