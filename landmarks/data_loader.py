#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器 - 用于加载鱼体关键点标注数据

支持的数据格式：
1. JSON格式：{"image_name.jpg": {"landmarks": [[x1,y1], [x2,y2]], "visibility": [1,1]}}
2. TXT格式：每行包含图像路径和关键点坐标
3. COCO格式：标准的COCO关键点格式
"""

import json
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2
from pathlib import Path


class FishLandmarkDataLoader:
    """鱼体关键点数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.landmark_names = ['body_center']
        self.num_landmarks = len(self.landmark_names)
    
    def load_from_json(self, json_path: str) -> Tuple[List[str], List[np.ndarray]]:
        """
        从JSON文件加载数据
        
        JSON格式：
        {
            "image1.jpg": {
                "landmarks": [[x1, y1], [x2, y2]],
                "visibility": [1, 1]
            },
            "image2.jpg": {
                "landmarks": [[x1, y1], [x2, y2]],
                "visibility": [1, 0]
            }
        }
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_paths = []
        landmarks_list = []
        
        for image_name, annotation in data.items():
            # 构建完整图像路径
            image_path = self.data_dir / "images" / image_name
            if not image_path.exists():
                print(f"警告: 图像文件不存在: {image_path}")
                continue
            
            all_landmarks = np.array(annotation['landmarks'], dtype=np.float32)
            all_visibility = np.array(annotation.get('visibility', [1] * len(all_landmarks)))
            
            # 只使用身体中心（第二个关键点，索引为1）
            if len(all_landmarks) >= 2:
                landmarks = all_landmarks[1:2]  # 只取身体中心
                visibility = all_visibility[1:2]  # 只取身体中心的可见性
            else:
                print(f"警告: 图像 {image_name} 的关键点数量不足: {len(all_landmarks)} < 2")
                continue
            
            image_paths.append(str(image_path))
            landmarks_list.append(landmarks)
        
        print(f"从JSON加载了 {len(image_paths)} 个样本")
        return image_paths, landmarks_list
    
    def load_from_txt(self, txt_path: str) -> Tuple[List[str], List[np.ndarray]]:
        """
        从TXT文件加载数据
        
        TXT格式：
        每行: image_path x1 y1 x2 y2
        例如: images/fish1.jpg 100 150 200 250
        """
        image_paths = []
        landmarks_list = []
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) != 5:  # image_path + 4 coordinates
                    print(f"警告: 第{line_num}行格式不正确: {line}")
                    continue
                
                image_path = self.data_dir / parts[0]
                if not image_path.exists():
                    print(f"警告: 图像文件不存在: {image_path}")
                    continue
                
                try:
                    x1, y1, x2, y2 = map(float, parts[1:5])
                    landmarks = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    
                    image_paths.append(str(image_path))
                    landmarks_list.append(landmarks)
                except ValueError:
                    print(f"警告: 第{line_num}行坐标格式错误: {line}")
                    continue
        
        print(f"从TXT加载了 {len(image_paths)} 个样本")
        return image_paths, landmarks_list
    
    def load_from_coco(self, json_path: str) -> Tuple[List[str], List[np.ndarray]]:
        """
        从COCO格式JSON文件加载数据
        
        COCO格式包含categories和annotations
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 构建图像ID到文件名的映射
        images = {img['id']: img for img in coco_data['images']}
        
        # 构建类别ID到关键点名称的映射
        categories = {cat['id']: cat for cat in coco_data['categories']}
        
        image_paths = []
        landmarks_list = []
        
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in images:
                continue
            
            image_info = images[image_id]
            image_name = image_info['file_name']
            image_path = self.data_dir / "images" / image_name
            
            if not image_path.exists():
                print(f"警告: 图像文件不存在: {image_path}")
                continue
            
            # 提取关键点
            keypoints = annotation.get('keypoints', [])
            if len(keypoints) != self.num_landmarks * 3:  # x, y, visibility
                print(f"警告: 图像 {image_name} 的关键点数量不正确")
                continue
            
            landmarks = []
            for i in range(0, len(keypoints), 3):
                x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
                if v > 0:  # 可见
                    landmarks.append([x, y])
                else:  # 不可见，使用默认位置
                    landmarks.append([0, 0])
            
            landmarks = np.array(landmarks, dtype=np.float32)
            
            image_paths.append(str(image_path))
            landmarks_list.append(landmarks)
        
        print(f"从COCO格式加载了 {len(image_paths)} 个样本")
        return image_paths, landmarks_list
    
    def validate_data(self, image_paths: List[str], landmarks_list: List[np.ndarray]) -> Tuple[List[str], List[np.ndarray]]:
        """验证和清理数据"""
        valid_image_paths = []
        valid_landmarks = []
        
        for i, (image_path, landmarks) in enumerate(zip(image_paths, landmarks_list)):
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                print(f"跳过不存在的图像: {image_path}")
                continue
            
            # 检查关键点格式
            if landmarks.shape != (self.num_landmarks, 2):
                print(f"跳过格式错误的关键点: {image_path}")
                continue
            
            # 检查关键点是否在合理范围内
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"跳过无法读取的图像: {image_path}")
                    continue
                
                h, w = img.shape[:2]
                # 检查关键点是否在图像范围内
                valid_points = True
                for landmark in landmarks:
                    x, y = landmark
                    if x < 0 or x >= w or y < 0 or y >= h:
                        print(f"警告: 关键点超出图像范围: {image_path}, 点: ({x}, {y}), 图像尺寸: ({w}, {h})")
                        # 可以选择跳过或修正
                        # valid_points = False
                        # break
                
                if valid_points:
                    valid_image_paths.append(image_path)
                    valid_landmarks.append(landmarks)
                    
            except Exception as e:
                print(f"跳过有问题的图像: {image_path}, 错误: {e}")
                continue
        
        print(f"数据验证完成: {len(valid_image_paths)}/{len(image_paths)} 个有效样本")
        return valid_image_paths, valid_landmarks
    
    def get_statistics(self, landmarks_list: List[np.ndarray]) -> Dict:
        """获取数据统计信息"""
        if not landmarks_list:
            return {}
        
        all_landmarks = np.concatenate(landmarks_list, axis=0)
        
        stats = {
            'total_samples': len(landmarks_list),
            'total_landmarks': len(all_landmarks),
            'landmark_names': self.landmark_names,
            'x_range': [float(np.min(all_landmarks[:, 0])), float(np.max(all_landmarks[:, 0]))],
            'y_range': [float(np.min(all_landmarks[:, 1])), float(np.max(all_landmarks[:, 1]))],
            'x_mean': float(np.mean(all_landmarks[:, 0])),
            'y_mean': float(np.mean(all_landmarks[:, 1])),
            'x_std': float(np.std(all_landmarks[:, 0])),
            'y_std': float(np.std(all_landmarks[:, 1]))
        }
        
        return stats
    
    def save_processed_data(self, image_paths: List[str], landmarks_list: List[np.ndarray], 
                           output_path: str, format: str = 'json'):
        """保存处理后的数据"""
        if format == 'json':
            data = {}
            for image_path, landmarks in zip(image_paths, landmarks_list):
                image_name = os.path.basename(image_path)
                data[image_name] = {
                    'landmarks': landmarks.tolist(),
                    'visibility': [1] * self.num_landmarks  # 假设所有点都可见
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for image_path, landmarks in zip(image_paths, landmarks_list):
                    # 使用相对路径
                    rel_path = os.path.relpath(image_path, self.data_dir)
                    coords = ' '.join([f"{x} {y}" for x, y in landmarks])
                    f.write(f"{rel_path} {coords}\n")
        
        print(f"数据已保存到: {output_path}")


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='鱼体关键点数据加载器')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--input_file', type=str, required=True, help='输入文件路径')
    parser.add_argument('--format', type=str, choices=['json', 'txt', 'coco'], default='json', help='数据格式')
    parser.add_argument('--output_file', type=str, help='输出文件路径（可选）')
    parser.add_argument('--stats', action='store_true', help='显示数据统计信息')
    
    args = parser.parse_args()
    
    # 创建数据加载器
    loader = FishLandmarkDataLoader(args.data_dir)
    
    # 加载数据
    if args.format == 'json':
        image_paths, landmarks_list = loader.load_from_json(args.input_file)
    elif args.format == 'txt':
        image_paths, landmarks_list = loader.load_from_txt(args.input_file)
    elif args.format == 'coco':
        image_paths, landmarks_list = loader.load_from_coco(args.input_file)
    
    # 验证数据
    image_paths, landmarks_list = loader.validate_data(image_paths, landmarks_list)
    
    # 显示统计信息
    if args.stats:
        stats = loader.get_statistics(landmarks_list)
        print("\n数据统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # 保存处理后的数据
    if args.output_file:
        loader.save_processed_data(image_paths, landmarks_list, args.output_file, args.format)
    
    print(f"\n成功加载 {len(image_paths)} 个样本")


if __name__ == "__main__":
    main()



