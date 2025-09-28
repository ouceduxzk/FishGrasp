#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理bbox和关键点数据

从JSON标注文件中提取bbox，裁剪图像，并调整关键点坐标

使用示例:
    # 基本用法
    python3 process_bbox_landmark_data.py --data_dir ../final_data/landmarks_9.18 --output_dir ./process_data --create_annotations
    
    # 查看帮助和使用示例
    python process_bbox_landmark_data.py --help
    python process_bbox_landmark_data.py  # 无参数时显示使用示例

输入数据格式:
    - 图像文件: JPG, PNG, BMP, TIFF等格式
    - 标注文件: JSON格式，包含关键点和bbox信息
    - 目录结构: data_dir/images/{train,val}/ 和 data_dir/labels/{train,val}/

JSON标注文件示例:
    # LabelMe格式
    {
        "shapes": [
            {"label": "头", "shape_type": "point", "points": [[100, 50]]},
            {"label": "身体", "shape_type": "point", "points": [[100, 150]]},
            {"label": "尾部", "shape_type": "point", "points": [[100, 250]]},
            {"label": "鱿鱼", "shape_type": "rectangle", "points": [[50, 25], [150, 275]]}
        ]
    }
    
    # 原始格式（向后兼容）
    {
        "头部": [100, 50],      # 头部中心点坐标
        "身体": [100, 150],     # 身体中心点坐标
        "bbox": [50, 25, 150, 175]  # 边界框 [x1, y1, x2, y2]
    }

输出文件:
    - 裁剪后的图像文件
    - 关键点numpy数组文件
    - 训练用标注JSON文件
    - 处理摘要和统计信息
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import shutil


class BboxLandmarkProcessor:
    """
    处理bbox和关键点数据的类
    
    使用示例:
        # 创建处理器实例
        processor = BboxLandmarkProcessor('./data', './output')
        
        # 处理所有数据
        all_data = processor.process_all_data()
        
        # 创建训练标注文件
        processor.create_training_annotations(all_data)
    """
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        初始化处理器
        
        Args:
            data_dir: 数据根目录 (包含images和labels子目录)
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录结构
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "landmarks").mkdir(exist_ok=True)
        
        # 中文关键点名称映射 - 只保留头部和身体
        self.landmark_mapping = {
            '头部': 'head_center',
            '身体': 'body_center',
            '头': 'head_center',
            '身体': 'body_center',
            'head': 'head_center',
            'body': 'body_center',
            '头部中心': 'head_center',
            '身体中心': 'body_center'
        }
        
        print(f"✅ 数据处理器初始化完成")
        print(f"   数据目录: {self.data_dir}")
        print(f"   输出目录: {self.output_dir}")
    
    def load_json_annotation(self, json_path: Path) -> Optional[Dict]:
        """加载JSON标注文件"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ 无法加载JSON文件 {json_path}: {e}")
            return None
    
    def extract_landmarks_from_annotation(self, annotation: Dict) -> List[Tuple[str, List[float]]]:
        """
        从标注中提取关键点（只保留头部和身体）
        
        Args:
            annotation: JSON标注数据
            
        Returns:
            List of (landmark_name, [x, y]) tuples
        """
        landmarks = []
        
        # 检查是否是LabelMe格式
        if 'shapes' in annotation:
            # LabelMe格式处理
            for shape in annotation['shapes']:
                if shape.get('shape_type') == 'point':
                    label = shape.get('label', '')
                    points = shape.get('points', [])
                    
                    # 只处理头部和身体关键点
                    if label in self.landmark_mapping and len(points) > 0:
                        try:
                            x, y = float(points[0][0]), float(points[0][1])
                            mapped_name = self.landmark_mapping[label]
                            landmarks.append((mapped_name, [x, y]))
                        except (ValueError, IndexError, TypeError):
                            print(f"⚠️  无效的关键点坐标: {label} = {points}")
        
        else:
            # 原始格式处理（向后兼容）
            for key, value in annotation.items():
                if isinstance(value, dict):
                    # 检查是否是关键点标注
                    for landmark_name, coords in value.items():
                        # 只处理头部和身体关键点
                        if landmark_name in self.landmark_mapping:
                            if isinstance(coords, list) and len(coords) >= 2:
                                # 确保坐标是数字
                                try:
                                    x, y = float(coords[0]), float(coords[1])
                                    mapped_name = self.landmark_mapping[landmark_name]
                                    landmarks.append((mapped_name, [x, y]))
                                except (ValueError, IndexError):
                                    print(f"⚠️  无效的关键点坐标: {landmark_name} = {coords}")
        
        return landmarks
    
    def filter_landmarks_for_bbox(self, landmarks: List[Tuple[str, List[float]]], 
                                 bbox: List[float]) -> List[Tuple[str, List[float]]]:
        """
        为特定bbox过滤关键点，选择最相关的头部和身体关键点
        
        策略：
        1. 优先选择在bbox内部的关键点
        2. 如果没有内部关键点，选择距离bbox中心最近的
        3. 每个类型（头部/身体）最多选择一个关键点
        
        Args:
            landmarks: 所有关键点列表
            bbox: [x1, y1, x2, y2]
            
        Returns:
            过滤后的关键点列表，每个类型最多一个
        """
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # 按类型分组关键点
        head_landmarks = []
        body_landmarks = []
        
        for name, coord in landmarks:
            if name == 'head_center':
                head_landmarks.append(coord)
            elif name == 'body_center':
                body_landmarks.append(coord)
        
        filtered_landmarks = []
        
        # 选择最佳头部关键点
        if head_landmarks:
            best_head = self._select_best_landmark(head_landmarks, bbox, bbox_center_x, bbox_center_y)
            filtered_landmarks.append(('head_center', best_head))
        
        # 选择最佳身体关键点
        if body_landmarks:
            best_body = self._select_best_landmark(body_landmarks, bbox, bbox_center_x, bbox_center_y)
            filtered_landmarks.append(('body_center', best_body))
        
        return filtered_landmarks
    
    def _select_best_landmark(self, landmarks: List[List[float]], bbox: List[float], 
                            bbox_center_x: float, bbox_center_y: float) -> List[float]:
        """
        从关键点列表中选择最佳的一个
        
        Args:
            landmarks: 关键点坐标列表
            bbox: [x1, y1, x2, y2]
            bbox_center_x: bbox中心x坐标
            bbox_center_y: bbox中心y坐标
            
        Returns:
            最佳关键点坐标
        """
        x1, y1, x2, y2 = bbox
        
        # 首先尝试找到在bbox内部的关键点
        inside_landmarks = []
        for coord in landmarks:
            x, y = coord
            if x1 <= x <= x2 and y1 <= y <= y2:
                inside_landmarks.append(coord)
        
        # 如果有内部关键点，选择距离中心最近的
        if inside_landmarks:
            return min(inside_landmarks, 
                      key=lambda coord: ((coord[0] - bbox_center_x) ** 2 + 
                                       (coord[1] - bbox_center_y) ** 2) ** 0.5)
        
        # 如果没有内部关键点，选择距离中心最近的
        return min(landmarks, 
                  key=lambda coord: ((coord[0] - bbox_center_x) ** 2 + 
                                   (coord[1] - bbox_center_y) ** 2) ** 0.5)
    
    def extract_bbox_from_annotation(self, annotation: Dict) -> Optional[List[float]]:
        """
        从标注中提取bbox
        
        Args:
            annotation: JSON标注数据
            
        Returns:
            [x1, y1, x2, y2] 或 None
        """
        # 检查是否是LabelMe格式
        if 'shapes' in annotation:
            # LabelMe格式处理
            for shape in annotation['shapes']:
                if shape.get('shape_type') == 'rectangle':
                    points = shape.get('points', [])
                    if len(points) >= 2:
                        try:
                            # LabelMe格式：points[0]是左上角，points[1]是右下角
                            x1, y1 = float(points[0][0]), float(points[0][1])
                            x2, y2 = float(points[1][0]), float(points[1][1])
                            
                            # 确保坐标顺序正确
                            if x1 > x2:
                                x1, x2 = x2, x1
                            if y1 > y2:
                                y1, y2 = y2, y1
                                
                            return [x1, y1, x2, y2]
                        except (ValueError, IndexError, TypeError):
                            print(f"⚠️  无效的边界框坐标: {points}")
                            continue
        
        # 原始格式处理（向后兼容）
        bbox_fields = ['bbox', 'bounding_box', 'rect', 'rectangle', 'box']
        
        for field in bbox_fields:
            if field in annotation:
                bbox = annotation[field]
                if isinstance(bbox, list) and len(bbox) >= 4:
                    try:
                        x1, y1, x2, y2 = map(float, bbox[:4])
                        return [x1, y1, x2, y2]
                    except (ValueError, IndexError):
                        continue
        
        # 如果没有找到bbox，尝试从关键点计算
        landmarks = self.extract_landmarks_from_annotation(annotation)
        if len(landmarks) >= 2:
            # 使用关键点计算bbox
            all_x = [coord[0] for _, coord in landmarks]
            all_y = [coord[1] for _, coord in landmarks]
            
            x1, x2 = min(all_x), max(all_x)
            y1, y2 = min(all_y), max(all_y)
            
            # 添加一些边距
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = x2 + margin
            y2 = y2 + margin
            
            return [x1, y1, x2, y2]
        
        return None
    
    def extract_all_bboxes_from_annotation(self, annotation: Dict) -> List[List[float]]:
        """
        从标注中提取所有bbox
        
        Args:
            annotation: JSON标注数据
            
        Returns:
            List of [x1, y1, x2, y2] bboxes
        """
        bboxes = []
        
        # 检查是否是LabelMe格式
        if 'shapes' in annotation:
            # LabelMe格式处理
            for shape in annotation['shapes']:
                if shape.get('shape_type') == 'rectangle':
                    points = shape.get('points', [])
                    if len(points) >= 2:
                        try:
                            # LabelMe格式：points[0]是左上角，points[1]是右下角
                            x1, y1 = float(points[0][0]), float(points[0][1])
                            x2, y2 = float(points[1][0]), float(points[1][1])
                            
                            # 确保坐标顺序正确
                            if x1 > x2:
                                x1, x2 = x2, x1
                            if y1 > y2:
                                y1, y2 = y2, y1
                                
                            bboxes.append([x1, y1, x2, y2])
                        except (ValueError, IndexError, TypeError):
                            print(f"⚠️  无效的边界框坐标: {points}")
                            continue
        
        # 原始格式处理（向后兼容）
        bbox_fields = ['bbox', 'bounding_box', 'rect', 'rectangle', 'box']
        
        for field in bbox_fields:
            if field in annotation:
                bbox = annotation[field]
                if isinstance(bbox, list) and len(bbox) >= 4:
                    try:
                        x1, y1, x2, y2 = map(float, bbox[:4])
                        bboxes.append([x1, y1, x2, y2])
                    except (ValueError, IndexError):
                        continue
        
        # 如果没有找到bbox，尝试从关键点计算
        if not bboxes:
            landmarks = self.extract_landmarks_from_annotation(annotation)
            if len(landmarks) >= 2:
                # 使用关键点计算bbox
                all_x = [coord[0] for _, coord in landmarks]
                all_y = [coord[1] for _, coord in landmarks]
                
                x1, x2 = min(all_x), max(all_x)
                y1, y2 = min(all_y), max(all_y)
                
                # 添加一些边距
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = x2 + margin
                y2 = y2 + margin
                
                bboxes.append([x1, y1, x2, y2])
        
        return bboxes
    
    def crop_image_and_adjust_landmarks(self, image: np.ndarray, bbox: List[float], 
                                      landmarks: List[Tuple[str, List[float]]]) -> Tuple[np.ndarray, List[Tuple[str, List[float]]]]:
        """
        裁剪图像并调整关键点坐标
        
        Args:
            image: 原始图像
            bbox: [x1, y1, x2, y2]
            landmarks: [(name, [x, y]), ...]
            
        Returns:
            (cropped_image, adjusted_landmarks)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # 确保bbox在图像范围内
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # 裁剪图像
        cropped_image = image[y1:y2, x1:x2]
        
        # 调整关键点坐标
        adjusted_landmarks = []
        for name, coord in landmarks:
            x, y = coord
            # 减去bbox的左上角坐标
            new_x = x - x1
            new_y = y - y1
            adjusted_landmarks.append((name, [new_x, new_y]))
        
        return cropped_image, adjusted_landmarks
    
    def visualize_landmarks_on_cropped_image(self, cropped_image: np.ndarray, 
                                           landmarks: List[Tuple[str, List[float]]]) -> np.ndarray:
        """
        在裁剪后的图像上可视化关键点
        
        Args:
            cropped_image: 裁剪后的图像
            landmarks: [(name, [x, y]), ...]
            
        Returns:
            带有关键点可视化的图像
        """
        vis_image = cropped_image.copy()
        
        # 定义关键点颜色
        colors = {
            'head_center': (0, 255, 0),    # 绿色 - 头部
            'body_center': (255, 0, 0),    # 蓝色 - 身体
        }
        
        # 绘制关键点
        for name, coord in landmarks:
            x, y = int(coord[0]), int(coord[1])
            color = colors.get(name, (255, 255, 255))  # 默认白色
            
            # 绘制圆点
            cv2.circle(vis_image, (x, y), 5, color, -1)
            
            # 绘制标签
            label = name.replace('_center', '').replace('_', ' ').title()
            cv2.putText(vis_image, label, (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image
    
    def process_single_image(self, image_path: Path, json_path: Path, 
                           output_prefix: str) -> List[Dict]:
        """
        处理单个图像文件，支持多个bbox
        
        Args:
            image_path: 图像文件路径
            json_path: JSON标注文件路径
            output_prefix: 输出文件前缀
            
        Returns:
            List of processed data dictionaries (每个bbox一个结果)
        """
        # 加载图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 无法加载图像: {image_path}")
            return []
        
        # 加载标注
        annotation = self.load_json_annotation(json_path)
        if annotation is None:
            return []
        
        # 提取关键点
        landmarks = self.extract_landmarks_from_annotation(annotation)
        if len(landmarks) < 2:
            print(f"⚠️  图像 {image_path.name} 的关键点数量不足: {len(landmarks)}")
            return []
        
        # 提取所有bbox
        all_bboxes = self.extract_all_bboxes_from_annotation(annotation)
        if not all_bboxes:
            print(f"⚠️  图像 {image_path.name} 没有找到有效的bbox")
            return []
        
        results = []
        
        # 处理每个bbox
        for i, bbox in enumerate(all_bboxes):
            # 为当前bbox过滤关键点，选择最相关的头部和身体关键点
            filtered_landmarks = self.filter_landmarks_for_bbox(landmarks, bbox)
            
            if len(filtered_landmarks) < 2:
                print(f"⚠️  bbox {i+1} 过滤后关键点不足: {len(filtered_landmarks)}")
                continue
            
            # 裁剪图像并调整关键点
            cropped_image, adjusted_landmarks = self.crop_image_and_adjust_landmarks(
                image, bbox, filtered_landmarks
            )
            
            # 检查关键点是否在裁剪后的图像范围内
            valid_landmarks = []
            for name, coord in adjusted_landmarks:
                x, y = coord
                h, w = cropped_image.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    valid_landmarks.append((name, coord))
            
            if len(valid_landmarks) < 2:
                print(f"⚠️  bbox {i+1} 在裁剪后图像中有效关键点不足: {len(valid_landmarks)}")
                continue
            
            # 保存裁剪后的图像
            bbox_output_prefix = f"{output_prefix}_bbox_{i+1}"
            output_image_path = self.output_dir / "images" / f"{bbox_output_prefix}.jpg"
            cv2.imwrite(str(output_image_path), cropped_image)
            
            # 创建可视化图像
            vis_image = self.visualize_landmarks_on_cropped_image(cropped_image, valid_landmarks)
            vis_output_path = self.output_dir / "images" / f"{bbox_output_prefix}_vis.jpg"
            cv2.imwrite(str(vis_output_path), vis_image)
            
            # 准备关键点数据（只保留头部和身体）
            landmark_data = {
                'head_center': None,
                'body_center': None
            }
            
            for name, coord in valid_landmarks:
                if name in landmark_data:
                    landmark_data[name] = coord
            
            # 检查是否有关键点缺失（至少需要头部和身体）
            required_landmarks = ['head_center', 'body_center']
            missing_landmarks = [name for name in required_landmarks if landmark_data[name] is None]
            if missing_landmarks:
                print(f"⚠️  bbox {i+1} 缺少必需关键点: {missing_landmarks}")
                continue
            
            # 保存关键点数据（按顺序：头部、身体）
            landmark_list = []
            for landmark_name in ['head_center', 'body_center']:
                if landmark_data[landmark_name] is not None:
                    landmark_list.append(landmark_data[landmark_name])
            
            landmark_array = np.array(landmark_list, dtype=np.float32)
            
            output_landmark_path = self.output_dir / "landmarks" / f"{bbox_output_prefix}.npy"
            np.save(str(output_landmark_path), landmark_array)
            
            # 返回处理结果
            result = {
                'image_path': str(output_image_path),
                'vis_image_path': str(vis_output_path),
                'landmark_path': str(output_landmark_path),
                'landmarks': landmark_data,
                'bbox': bbox,
                'bbox_index': i + 1,
                'original_image': str(image_path),
                'original_json': str(json_path)
            }
            
            results.append(result)
            print(f"✅ 处理bbox {i+1}: {bbox}, 关键点: {len(valid_landmarks)} 个")
        
        return results
    
    def process_dataset_split(self, split: str) -> List[Dict]:
        """
        处理数据集的一个分割（train或val）
        
        Args:
            split: 'train' 或 'val'
            
        Returns:
            List of processed data dictionaries
        """
        print(f"\n🔄 处理 {split} 数据集...")
        
        # 检查是否存在 train/val 子目录结构
        images_dir = self.data_dir / "images" / split
        labels_dir = self.data_dir / "labels" / split
        
        # 如果不存在子目录，使用直接的 images/ 和 labels/ 目录
        if not images_dir.exists():
            images_dir = self.data_dir / "images"
            labels_dir = self.data_dir / "labels"
            print(f"📂 使用统一目录结构: {images_dir}, {labels_dir}")
        
        if not images_dir.exists():
            print(f"❌ 图像目录不存在: {images_dir}")
            return []
        
        if not labels_dir.exists():
            print(f"❌ 标注目录不存在: {labels_dir}")
            return []
        
        # 获取所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        print(f"   找到 {len(image_files)} 个图像文件")
        
        processed_data = []
        success_count = 0
        
        for image_path in image_files:
            # 查找对应的JSON文件
            json_name = image_path.stem + '.json'
            json_path = labels_dir / json_name
            
            if not json_path.exists():
                print(f"⚠️  找不到对应的JSON文件: {json_path}")
                continue
            
            # 处理图像
            output_prefix = f"{split}_{image_path.stem}"
            results = self.process_single_image(image_path, json_path, output_prefix)
            
            if results:
                processed_data.extend(results)
                success_count += 1
            
            if success_count % 10 == 0:
                print(f"   已处理: {success_count}/{len(image_files)}")
        
        print(f"✅ {split} 数据集处理完成: {success_count}/{len(image_files)} 成功")
        return processed_data
    
    def process_all_data(self) -> Dict[str, List[Dict]]:
        """处理所有数据"""
        print("🚀 开始处理所有数据...")
        
        all_data = {}
        
        # 检查是否存在 train/val 子目录结构
        train_images_dir = self.data_dir / "images" / "train"
        val_images_dir = self.data_dir / "images" / "val"
        
        if train_images_dir.exists() and val_images_dir.exists():
            # 标准格式：处理训练集和验证集
            print("📂 检测到标准目录结构 (train/val)")
            train_data = self.process_dataset_split('train')
            all_data['train'] = train_data
            
            val_data = self.process_dataset_split('val')
            all_data['val'] = val_data
        else:
            # 简化格式：处理所有数据作为训练集
            print("📂 检测到简化目录结构，将所有数据作为训练集")
            train_data = self.process_dataset_split('train')  # 这会使用统一的 images/ 和 labels/ 目录
            all_data['train'] = train_data
            all_data['val'] = []  # 验证集为空
        
        # 保存处理结果摘要
        self.save_processing_summary(all_data)
        
        return all_data
    
    def save_processing_summary(self, all_data: Dict[str, List[Dict]]):
        """保存处理结果摘要"""
        summary = {
            'total_samples': sum(len(data) for data in all_data.values()),
            'train_samples': len(all_data.get('train', [])),
            'val_samples': len(all_data.get('val', [])),
            'output_directory': str(self.output_dir),
            'landmark_names': ['head_center', 'body_center', 'tail_center'],
            'processing_date': str(Path().cwd())
        }
        
        # 保存摘要
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存详细的数据列表
        data_list_path = self.output_dir / "data_list.json"
        with open(data_list_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 处理摘要:")
        print(f"   总样本数: {summary['total_samples']}")
        print(f"   训练样本: {summary['train_samples']}")
        print(f"   验证样本: {summary['val_samples']}")
        print(f"   摘要文件: {summary_path}")
        print(f"   数据列表: {data_list_path}")
    
    def create_training_annotations(self, all_data: Dict[str, List[Dict]]):
        """创建训练用的标注文件"""
        print("\n📝 创建训练标注文件...")
        
        # 创建JSON格式的标注文件
        for split, data in all_data.items():
            annotations = {}
            
            for item in data:
                image_name = Path(item['image_path']).name
                
                # 构建关键点列表（按顺序：头部、身体）
                landmarks_list = []
                visibility_list = []
                
                for landmark_name in ['head_center', 'body_center']:
                    if item['landmarks'].get(landmark_name) is not None:
                        landmarks_list.append(item['landmarks'][landmark_name])
                        visibility_list.append(1)  # 可见
                    else:
                        print(f"⚠️  缺少关键点 {landmark_name} 在 {item['image_path']}")
                        # 如果关键点缺失，跳过这个样本
                        continue
                
                annotations[image_name] = {
                    'landmarks': landmarks_list,
                    'visibility': visibility_list
                }
            
            # 保存标注文件
            annotation_path = self.output_dir / f"{split}_annotations.json"
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=2, ensure_ascii=False)
            
            print(f"   {split} 标注文件: {annotation_path} ({len(annotations)} 样本)")


def main():
    parser = argparse.ArgumentParser(
        description='处理bbox和关键点数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --data_dir ./data --output_dir ./output --create_annotations
  %(prog)s --data_dir ./raw_9.9_sum --output_dir ./landmarks/processed_9.9_sum
  %(prog)s --help  # 查看详细使用说明
        """
    )
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='数据根目录 (包含images和labels子目录)')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='输出目录 (将创建images和landmarks子目录)')
    parser.add_argument('--create_annotations', action='store_true', 
                       help='创建训练用的标注文件 (*_annotations.json)')
    
    args = parser.parse_args()
    
    # 检查数据目录
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 检查目录结构 - 支持两种格式
    # 格式1: images/train, images/val, labels/train, labels/val (标准格式)
    # 格式2: images/, labels/ (简化格式)
    required_dirs_standard = ['images/train', 'images/val', 'labels/train', 'labels/val']
    required_dirs_simple = ['images', 'labels']
    
    # 检查标准格式
    standard_format = all((data_dir / req_dir).exists() for req_dir in required_dirs_standard)
    # 检查简化格式
    simple_format = all((data_dir / req_dir).exists() for req_dir in required_dirs_simple)
    
    if not standard_format and not simple_format:
        print(f"❌ 目录结构不正确!")
        print(f"   支持的格式:")
        print(f"   格式1 (标准): {[str(data_dir / d) for d in required_dirs_standard]}")
        print(f"   格式2 (简化): {[str(data_dir / d) for d in required_dirs_simple]}")
        return
    
    if simple_format and not standard_format:
        print(f"📂 检测到简化目录结构，将处理所有数据")
    
    # 创建处理器
    processor = BboxLandmarkProcessor(args.data_dir, args.output_dir)
    
    # 处理数据
    all_data = processor.process_all_data()
    
    # 创建训练标注文件
    if args.create_annotations:
        processor.create_training_annotations(all_data)
    
    print("\n🎉 数据处理完成！")
    print(f"输出目录: {args.output_dir}")
    print("\n目录结构:")
    print("  images/     - 裁剪后的图像")
    print("  landmarks/  - 关键点numpy文件")
    print("  *_annotations.json - 训练用标注文件")


def print_usage_examples():
    """打印使用示例"""
    print("""
📖 使用示例:

1. 基本用法 - 处理数据并创建训练标注文件:
   python process_bbox_landmark_data.py \\
       --data_dir /path/to/your/data \\
       --output_dir /path/to/output \\
       --create_annotations

2. 仅处理数据，不创建训练标注文件:
   python process_bbox_landmark_data.py \\
       --data_dir /path/to/your/data \\
       --output_dir /path/to/output

3. 使用项目中的数据目录:
   python process_bbox_landmark_data.py \\
       --data_dir ./data \\
       --output_dir ./landmarks/processed_data \\
       --create_annotations

4. 使用raw_9.9_sum数据:
   python process_bbox_landmark_data.py \\
       --data_dir ./raw_9.9_sum \\
       --output_dir ./landmarks/processed_9.9_sum \\
       --create_annotations

📁 输入数据目录结构要求:
   data_dir/
   ├── images/
   │   ├── train/          # 训练图像
   │   └── val/            # 验证图像
   └── labels/
       ├── train/          # 训练标注JSON文件
       └── val/            # 验证标注JSON文件

📄 JSON标注文件格式示例:
   # LabelMe格式
   {
     "shapes": [
       {"label": "头", "shape_type": "point", "points": [[100, 50]]},
       {"label": "身体", "shape_type": "point", "points": [[100, 150]]},
       {"label": "尾部", "shape_type": "point", "points": [[100, 250]]},
       {"label": "鱿鱼", "shape_type": "rectangle", "points": [[50, 25], [150, 275]]}
     ]
   }
   
   # 原始格式（向后兼容）
   {
     "头部": [100, 50],
     "身体": [100, 150],
     "bbox": [50, 25, 150, 175]
   }

📤 输出目录结构:
   output_dir/
   ├── images/             # 裁剪后的图像
   ├── landmarks/          # 关键点numpy文件
   ├── train_annotations.json    # 训练标注文件
   ├── val_annotations.json      # 验证标注文件
   ├── processing_summary.json   # 处理摘要
   └── data_list.json           # 详细数据列表

🔧 程序功能:
   - 从JSON标注文件中提取bbox和关键点
   - 根据bbox裁剪图像
   - 调整关键点坐标到裁剪后的图像坐标系
   - 保存处理后的图像和关键点数据
   - 生成训练用的标注文件

⚠️  注意事项:
   - 确保输入目录包含完整的images和labels子目录
   - JSON文件中的关键点名称支持中文和英文
   - 如果JSON中没有bbox，程序会从关键点自动计算
   - 处理过程中会跳过无效或缺失关键点的样本
""")


if __name__ == "__main__":
    import sys
    
    # 如果没有参数，显示使用示例
    if len(sys.argv) == 1:
        print_usage_examples()
        sys.exit(0)
    
    # 如果有--help或-h参数，显示使用示例
    if '--help' in sys.argv or '-h' in sys.argv:
        print_usage_examples()
        sys.exit(0)
    
    main()
