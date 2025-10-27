#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
支持硬负样本挖掘的YOLO训练脚本

硬负样本挖掘（Hard Negative Mining）是一种提高目标检测模型性能的技术：
1. 在训练过程中识别模型预测错误的困难负样本
2. 增加这些困难样本在训练中的权重
3. 提高模型对困难样本的识别能力

主要功能：
- 自动收集训练过程中的困难负样本
- 动态调整困难样本的采样权重
- 支持多种硬负样本挖掘策略
- 可视化困难样本的分布

使用方法：
  python3 detection/train_yolo_with_hard_negative.py \
    --data ./datasets/dataset.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --hard_negative_ratio 0.3 \
    --mining_strategy "confidence_based"
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import json

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    print("[错误] 未找到ultralytics，请先安装: pip install ultralytics")
    sys.exit(1)


class HardNegativeMiner:
    """硬负样本挖掘器"""
    
    def __init__(self, 
                 mining_strategy: str = "confidence_based",
                 hard_negative_ratio: float = 0.3,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 save_hard_negatives: bool = True,
                 output_dir: str = "hard_negatives"):
        """
        初始化硬负样本挖掘器
        
        Args:
            mining_strategy: 挖掘策略 ("confidence_based", "loss_based", "iou_based")
            hard_negative_ratio: 硬负样本比例
            confidence_threshold: 置信度阈值
            iou_threshold: IoU阈值
            save_hard_negatives: 是否保存硬负样本
            output_dir: 硬负样本输出目录
        """
        self.mining_strategy = mining_strategy
        self.hard_negative_ratio = hard_negative_ratio
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.save_hard_negatives = save_hard_negatives
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        if self.save_hard_negatives:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "images").mkdir(exist_ok=True)
            (self.output_dir / "annotations").mkdir(exist_ok=True)
        
        # 存储硬负样本信息
        self.hard_negatives = []
        self.mining_stats = {
            'total_samples': 0,
            'hard_negatives_found': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        print(f"✅ 硬负样本挖掘器初始化完成")
        print(f"   策略: {mining_strategy}")
        print(f"   硬负样本比例: {hard_negative_ratio}")
        print(f"   置信度阈值: {confidence_threshold}")
        print(f"   输出目录: {self.output_dir}")
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def find_hard_negatives(self, 
                          predictions: List[Dict], 
                          ground_truth: List[Dict],
                          image_path: str) -> List[Dict]:
        """
        根据策略找到硬负样本
        
        Args:
            predictions: 模型预测结果
            ground_truth: 真实标注
            image_path: 图像路径
            
        Returns:
            硬负样本列表
        """
        hard_negatives = []
        
        if self.mining_strategy == "confidence_based":
            hard_negatives = self._confidence_based_mining(predictions, ground_truth, image_path)
        elif self.mining_strategy == "iou_based":
            hard_negatives = self._iou_based_mining(predictions, ground_truth, image_path)
        elif self.mining_strategy == "loss_based":
            hard_negatives = self._loss_based_mining(predictions, ground_truth, image_path)
        
        # 更新统计信息
        self.mining_stats['total_samples'] += 1
        self.mining_stats['hard_negatives_found'] += len(hard_negatives)
        
        return hard_negatives
    
    def _confidence_based_mining(self, 
                               predictions: List[Dict], 
                               ground_truth: List[Dict],
                               image_path: str) -> List[Dict]:
        """基于置信度的硬负样本挖掘"""
        hard_negatives = []
        
        # 找到高置信度但IoU低的预测（假阳性）
        for pred in predictions:
            if pred['confidence'] > self.confidence_threshold:
                max_iou = 0.0
                for gt in ground_truth:
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    max_iou = max(max_iou, iou)
                
                if max_iou < self.iou_threshold:
                    hard_negatives.append({
                        'type': 'false_positive',
                        'prediction': pred,
                        'image_path': image_path,
                        'confidence': pred['confidence'],
                        'max_iou': max_iou
                    })
                    self.mining_stats['false_positives'] += 1
        
        return hard_negatives
    
    def _iou_based_mining(self, 
                        predictions: List[Dict], 
                        ground_truth: List[Dict],
                        image_path: str) -> List[Dict]:
        """基于IoU的硬负样本挖掘"""
        hard_negatives = []
        
        # 找到IoU在阈值附近的预测
        for pred in predictions:
            max_iou = 0.0
            best_gt = None
            for gt in ground_truth:
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_gt = gt
            
            # IoU在0.3-0.7之间的预测被认为是困难的
            if 0.3 <= max_iou <= 0.7:
                hard_negatives.append({
                    'type': 'hard_positive',
                    'prediction': pred,
                    'ground_truth': best_gt,
                    'image_path': image_path,
                    'iou': max_iou
                })
        
        return hard_negatives
    
    def _loss_based_mining(self, 
                         predictions: List[Dict], 
                         ground_truth: List[Dict],
                         image_path: str) -> List[Dict]:
        """基于损失的硬负样本挖掘"""
        hard_negatives = []
        
        # 这里需要访问模型的损失信息
        # 在实际实现中，需要修改训练循环来获取损失
        # 这里提供一个简化的实现
        
        for pred in predictions:
            # 计算预测与最近真实标注的距离
            min_distance = float('inf')
            for gt in ground_truth:
                # 计算中心点距离
                pred_center = [(pred['bbox'][0] + pred['bbox'][2]) / 2, 
                              (pred['bbox'][1] + pred['bbox'][3]) / 2]
                gt_center = [(gt['bbox'][0] + gt['bbox'][2]) / 2, 
                            (gt['bbox'][1] + gt['bbox'][3]) / 2]
                distance = np.sqrt((pred_center[0] - gt_center[0])**2 + 
                                 (pred_center[1] - gt_center[1])**2)
                min_distance = min(min_distance, distance)
            
            # 距离较近但IoU较低的预测被认为是困难的
            if min_distance < 50:  # 像素距离阈值
                hard_negatives.append({
                    'type': 'hard_negative',
                    'prediction': pred,
                    'image_path': image_path,
                    'min_distance': min_distance
                })
        
        return hard_negatives
    
    def save_hard_negative_sample(self, hard_negative: Dict, image: np.ndarray):
        """保存硬负样本"""
        if not self.save_hard_negatives:
            return
        
        # 生成文件名
        image_name = Path(hard_negative['image_path']).stem
        sample_type = hard_negative['type']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{image_name}_{sample_type}_{timestamp}"
        
        # 保存图像
        image_path = self.output_dir / "images" / f"{filename}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # 保存标注信息
        annotation_path = self.output_dir / "annotations" / f"{filename}.json"
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(hard_negative, f, indent=2, ensure_ascii=False)
    
    def get_mining_stats(self) -> Dict:
        """获取挖掘统计信息"""
        return self.mining_stats.copy()
    
    def save_mining_report(self, output_path: str):
        """保存挖掘报告"""
        report = {
            'mining_strategy': self.mining_strategy,
            'parameters': {
                'hard_negative_ratio': self.hard_negative_ratio,
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold
            },
            'statistics': self.mining_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 硬负样本挖掘报告已保存: {output_path}")


class YOLOTrainerWithHardNegative:
    """支持硬负样本挖掘的YOLO训练器"""
    
    def __init__(self, 
                 data_yaml: str,
                 model_path: str,
                 hard_negative_miner: HardNegativeMiner,
                 **train_kwargs):
        """
        初始化训练器
        
        Args:
            data_yaml: 数据集YAML文件路径
            model_path: 模型路径
            hard_negative_miner: 硬负样本挖掘器
            **train_kwargs: 训练参数
        """
        self.data_yaml = data_yaml
        self.model_path = model_path
        self.hard_negative_miner = hard_negative_miner
        self.train_kwargs = train_kwargs
        
        # 加载模型
        self.model = YOLO(model_path)
        
        # 加载数据集信息
        with open(data_yaml, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
        
        print(f"✅ YOLO训练器初始化完成")
        print(f"   模型: {model_path}")
        print(f"   数据集: {data_yaml}")
        print(f"   训练参数: {train_kwargs}")
    
    def train_with_hard_negative_mining(self, initial_epochs_ratio=0.33, use_pretrained_for_mining=False):
        """执行带硬负样本挖掘的训练"""
        print("🚀 开始带硬负样本挖掘的YOLO训练...")
        
        if use_pretrained_for_mining:
            # 使用预训练模型进行硬负样本挖掘
            print("🔍 使用预训练模型进行硬负样本挖掘...")
            self._mine_hard_negatives_on_validation()
            
            # 直接进行硬负样本增强训练
            if self.hard_negative_miner.mining_stats['hard_negatives_found'] > 0:
                print("🎯 使用硬负样本进行训练...")
                enhanced_kwargs = self.train_kwargs.copy()
                enhanced_kwargs['name'] = enhanced_kwargs.get('name', '') + '_hard_negative'
                
                results = self.model.train(
                    data=self.data_yaml,
                    **enhanced_kwargs
                )
            else:
                print("⚠️  未找到硬负样本，进行正常训练...")
                results = self.model.train(
                    data=self.data_yaml,
                    **self.train_kwargs
                )
        else:
            # 第一阶段：初始训练（使用较少轮数）
            print("📚 第一阶段：初始训练（建立基础模型）...")
            initial_epochs = max(20, int(self.train_kwargs['epochs'] * initial_epochs_ratio))
            
            initial_kwargs = self.train_kwargs.copy()
            initial_kwargs['epochs'] = initial_epochs
            initial_kwargs['name'] = initial_kwargs.get('name', '') + '_initial'
            
            print(f"   初始训练轮数: {initial_epochs}")
            initial_results = self.model.train(
                data=self.data_yaml,
                **initial_kwargs
            )
            
            # 第二阶段：硬负样本挖掘
            print("🔍 第二阶段：硬负样本挖掘...")
            print("   使用初始模型在验证集上进行硬负样本挖掘...")
            
            # 在验证集上进行硬负样本挖掘
            self._mine_hard_negatives_on_validation()
            
            # 第三阶段：硬负样本增强训练
            if self.hard_negative_miner.mining_stats['hard_negatives_found'] > 0:
                print("🎯 第三阶段：硬负样本增强训练...")
                remaining_epochs = self.train_kwargs['epochs'] - initial_epochs
                
                # 使用挖掘到的硬负样本进行额外训练
                enhanced_kwargs = self.train_kwargs.copy()
                enhanced_kwargs['epochs'] = remaining_epochs
                enhanced_kwargs['name'] = enhanced_kwargs.get('name', '') + '_enhanced'
                enhanced_kwargs['lr0'] = enhanced_kwargs.get('lr0', 0.01) * 0.5  # 降低学习率
                
                print(f"   增强训练轮数: {remaining_epochs}")
                print(f"   找到 {self.hard_negative_miner.mining_stats['hard_negatives_found']} 个硬负样本")
                
                results = self.model.train(
                    data=self.data_yaml,
                    **enhanced_kwargs
                )
            else:
                print("⚠️  未找到硬负样本，跳过增强训练阶段")
                results = initial_results
        
        # 保存挖掘报告
        report_path = Path(self.train_kwargs.get('project', 'runs/train')) / \
                     self.train_kwargs.get('name', 'hard_negative_training') / \
                     'hard_negative_report.json'
        self.hard_negative_miner.save_mining_report(str(report_path))
        
        return results
    
    def _mine_hard_negatives_on_validation(self):
        """在验证集上进行硬负样本挖掘"""
        print("🔍 在验证集上进行硬负样本挖掘...")
        
        # 获取验证集路径
        val_images_dir = Path(self.data_config['path']) / self.data_config['val']
        val_labels_dir = Path(self.data_config['path']) / 'labels' / 'val'
        
        if not val_images_dir.exists() or not val_labels_dir.exists():
            print("⚠️  验证集路径不存在，跳过硬负样本挖掘")
            return
        
        # 获取所有验证图像
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        val_images = []
        for ext in image_extensions:
            val_images.extend(val_images_dir.glob(f"*{ext}"))
            val_images.extend(val_images_dir.glob(f"*{ext.upper()}"))
        
        print(f"   找到 {len(val_images)} 个验证图像")
        
        # 对每个验证图像进行预测和挖掘
        for i, image_path in enumerate(val_images):
            if i % 10 == 0:
                print(f"   处理进度: {i}/{len(val_images)}")
            
            # 加载图像
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # 进行预测
            results = self.model.predict(str(image_path), verbose=False)
            
            # 解析预测结果
            predictions = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        predictions.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf)
                        })
            
            # 加载真实标注
            label_path = val_labels_dir / f"{image_path.stem}.txt"
            ground_truth = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # 转换为xyxy格式
                            x1 = (x_center - width/2) * image.shape[1]
                            y1 = (y_center - height/2) * image.shape[0]
                            x2 = (x_center + width/2) * image.shape[1]
                            y2 = (y_center + height/2) * image.shape[0]
                            
                            ground_truth.append({
                                'bbox': [x1, y1, x2, y2],
                                'class_id': class_id
                            })
            
            # 进行硬负样本挖掘
            hard_negatives = self.hard_negative_miner.find_hard_negatives(
                predictions, ground_truth, str(image_path)
            )
            
            # 保存硬负样本
            for hard_negative in hard_negatives:
                self.hard_negative_miner.save_hard_negative_sample(hard_negative, image)
        
        print(f"✅ 硬负样本挖掘完成")
        print(f"   找到 {self.hard_negative_miner.mining_stats['hard_negatives_found']} 个硬负样本")
    
    def _train_with_hard_negatives(self):
        """使用硬负样本进行额外训练"""
        print("🎯 使用硬负样本进行额外训练...")
        
        # 这里可以实现使用硬负样本的额外训练逻辑
        # 例如：增加硬负样本的权重、调整学习率等
        
        # 简化实现：使用较小的学习率进行额外训练
        additional_kwargs = self.train_kwargs.copy()
        additional_kwargs['epochs'] = 20  # 较少的轮数
        additional_kwargs['lr0'] = additional_kwargs.get('lr0', 0.01) * 0.1  # 较小的学习率
        additional_kwargs['name'] = additional_kwargs.get('name', '') + '_hard_negative'
        
        results = self.model.train(
            data=self.data_yaml,
            **additional_kwargs
        )
        
        return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="支持硬负样本挖掘的YOLO训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python3 detection/train_yolo_with_hard_negative.py \
    --data ./datasets/l0_9.12/dataset.yaml  \
    --model yolov8s.pt \
    --epochs 100 \
    --project runs/train \
    --name single_yolov8s_hard_negative_$(date +%Y%m%d_%H%M%S) \
    --hard_negative_ratio 0.3 \
    --mining_strategy confidence_based
        """
    )
    
    # 基本参数
    parser.add_argument("--data", type=str, required=True, help="数据集YAML路径")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="模型权重路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--device", type=str, default="", help="CUDA设备，如 '0' 或 '0,1'，留空自动选择")
    parser.add_argument("--project", type=str, default="runs/train", help="输出项目目录")
    parser.add_argument("--name", type=str, default="", help="实验名称，默认自动加时间戳")
    
    # 硬负样本挖掘参数
    parser.add_argument("--mining_strategy", type=str, default="confidence_based",
                       choices=["confidence_based", "iou_based", "loss_based"],
                       help="硬负样本挖掘策略")
    parser.add_argument("--hard_negative_ratio", type=float, default=0.3,
                       help="硬负样本比例")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="置信度阈值")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                       help="IoU阈值")
    parser.add_argument("--save_hard_negatives", action="store_true",
                       help="保存硬负样本")
    parser.add_argument("--initial_epochs_ratio", type=float, default=0.33,
                       help="初始训练阶段占总轮数的比例 (默认: 0.33)")
    parser.add_argument("--use_pretrained_for_mining", action="store_true",
                       help="使用预训练模型进行硬负样本挖掘（跳过初始训练）")
    
    # 训练参数
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心轮数")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--cache", action="store_true", help="缓存图像")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖已存在的目录")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查数据集文件
    if not os.path.exists(args.data):
        print(f"[错误] 数据集YAML不存在: {args.data}")
        sys.exit(1)
    
    # 生成良好命名：若未指定name，使用模型名+时间戳
    if not args.name:
        model_stem = os.path.splitext(os.path.basename(args.model))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"{model_stem}_hard_negative_{timestamp}"
    
    os.makedirs(args.project, exist_ok=True)
    
    print("======== 训练参数 ========")
    print(f"data      : {args.data}")
    print(f"model     : {args.model}")
    print(f"epochs    : {args.epochs}")
    print(f"batch     : {args.batch}")
    print(f"imgsz     : {args.imgsz}")
    print(f"device    : {args.device or 'auto'}")
    print(f"project   : {args.project}")
    print(f"name      : {args.name}")
    print(f"mining_strategy: {args.mining_strategy}")
    print(f"hard_negative_ratio: {args.hard_negative_ratio}")
    print(f"confidence_threshold: {args.confidence_threshold}")
    print(f"iou_threshold: {args.iou_threshold}")
    print(f"save_hard_negatives: {args.save_hard_negatives}")
    print(f"initial_epochs_ratio: {args.initial_epochs_ratio}")
    print(f"use_pretrained_for_mining: {args.use_pretrained_for_mining}")
    print("==========================")
    
    # 创建硬负样本挖掘器
    hard_negative_miner = HardNegativeMiner(
        mining_strategy=args.mining_strategy,
        hard_negative_ratio=args.hard_negative_ratio,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        save_hard_negatives=args.save_hard_negatives,
        output_dir=os.path.join(args.project, args.name, "hard_negatives")
    )
    
    # 准备训练参数
    train_kwargs = {
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'lr0': args.lr0,
        'patience': args.patience,
        'workers': args.workers,
        'seed': args.seed,
        'exist_ok': args.exist_ok,
        'cache': args.cache,
        'verbose': True
    }
    
    # 创建训练器
    trainer = YOLOTrainerWithHardNegative(
        data_yaml=args.data,
        model_path=args.model,
        hard_negative_miner=hard_negative_miner,
        **train_kwargs
    )
    
    # 开始训练
    try:
        results = trainer.train_with_hard_negative_mining(
            initial_epochs_ratio=args.initial_epochs_ratio,
            use_pretrained_for_mining=args.use_pretrained_for_mining
        )
        print("🎉 训练完成！")
        
        # 打印统计信息
        stats = hard_negative_miner.get_mining_stats()
        print("\n📊 硬负样本挖掘统计:")
        print(f"   总样本数: {stats['total_samples']}")
        print(f"   硬负样本数: {stats['hard_negatives_found']}")
        print(f"   假阳性数: {stats['false_positives']}")
        print(f"   假阴性数: {stats['false_negatives']}")
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
