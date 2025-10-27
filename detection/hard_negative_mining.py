#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
硬负样本挖掘工具

这个模块提供了硬负样本挖掘的核心功能，可以集成到现有的YOLO训练流程中。

主要功能：
1. 识别训练过程中的困难样本
2. 动态调整样本权重
3. 生成硬负样本报告
4. 可视化困难样本分布

使用方法：
    from detection.hard_negative_mining import HardNegativeMiner
    
    miner = HardNegativeMiner()
    hard_negatives = miner.find_hard_negatives(predictions, ground_truth)
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt


class HardNegativeMiner:
    """硬负样本挖掘器"""
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 hard_negative_ratio: float = 0.3,
                 save_samples: bool = False,
                 output_dir: str = "hard_negatives"):
        """
        初始化硬负样本挖掘器
        
        Args:
            confidence_threshold: 置信度阈值
            iou_threshold: IoU阈值
            hard_negative_ratio: 硬负样本比例
            save_samples: 是否保存困难样本
            output_dir: 输出目录
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.hard_negative_ratio = hard_negative_ratio
        self.save_samples = save_samples
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        if self.save_samples:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "images").mkdir(exist_ok=True)
            (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_predictions': 0,
            'hard_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'confidence_distribution': [],
            'iou_distribution': []
        }
        
        print(f"✅ 硬负样本挖掘器初始化完成")
        print(f"   置信度阈值: {confidence_threshold}")
        print(f"   IoU阈值: {iou_threshold}")
        print(f"   硬负样本比例: {hard_negative_ratio}")
    
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
                          image_path: Optional[str] = None) -> List[Dict]:
        """
        找到硬负样本
        
        Args:
            predictions: 模型预测结果 [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'class': int}]
            ground_truth: 真实标注 [{'bbox': [x1,y1,x2,y2], 'class': int}]
            image_path: 图像路径（可选）
            
        Returns:
            硬负样本列表
        """
        hard_negatives = []
        
        # 更新统计信息
        self.stats['total_predictions'] += len(predictions)
        
        # 为每个预测找到最佳匹配的真实标注
        matched_gt = set()
        
        for pred in predictions:
            best_iou = 0.0
            best_gt_idx = -1
            
            # 找到IoU最高的真实标注
            for i, gt in enumerate(ground_truth):
                if i in matched_gt:
                    continue
                    
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # 记录IoU分布
            self.stats['iou_distribution'].append(best_iou)
            self.stats['confidence_distribution'].append(pred['confidence'])
            
            # 判断是否为硬负样本
            if pred['confidence'] > self.confidence_threshold:
                if best_iou < self.iou_threshold:
                    # 高置信度但低IoU -> 假阳性（硬负样本）
                    hard_negatives.append({
                        'type': 'false_positive',
                        'prediction': pred,
                        'ground_truth': ground_truth[best_gt_idx] if best_gt_idx >= 0 else None,
                        'iou': best_iou,
                        'confidence': pred['confidence'],
                        'image_path': image_path
                    })
                    self.stats['false_positives'] += 1
                    self.stats['hard_negatives'] += 1
                else:
                    # 高置信度高IoU -> 真阳性
                    self.stats['true_positives'] += 1
                    if best_gt_idx >= 0:
                        matched_gt.add(best_gt_idx)
            else:
                if best_iou >= self.iou_threshold:
                    # 低置信度高IoU -> 假阴性
                    hard_negatives.append({
                        'type': 'false_negative',
                        'prediction': pred,
                        'ground_truth': ground_truth[best_gt_idx] if best_gt_idx >= 0 else None,
                        'iou': best_iou,
                        'confidence': pred['confidence'],
                        'image_path': image_path
                    })
                    self.stats['false_negatives'] += 1
                    self.stats['hard_negatives'] += 1
        
        return hard_negatives
    
    def analyze_difficulty_distribution(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        分析困难样本的分布
        
        Args:
            predictions: 预测结果
            ground_truth: 真实标注
            
        Returns:
            分析结果字典
        """
        analysis = {
            'confidence_ranges': {
                'high_conf_low_iou': 0,    # 高置信度低IoU
                'high_conf_high_iou': 0,   # 高置信度高IoU
                'low_conf_low_iou': 0,     # 低置信度低IoU
                'low_conf_high_iou': 0     # 低置信度高IoU
            },
            'iou_ranges': {
                'very_low': 0,    # IoU < 0.3
                'low': 0,         # 0.3 <= IoU < 0.5
                'medium': 0,      # 0.5 <= IoU < 0.7
                'high': 0         # IoU >= 0.7
            },
            'confidence_stats': {},
            'iou_stats': {}
        }
        
        ious = []
        confidences = []
        
        for pred in predictions:
            best_iou = 0.0
            for gt in ground_truth:
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                best_iou = max(best_iou, iou)
            
            ious.append(best_iou)
            confidences.append(pred['confidence'])
            
            # 分类置信度和IoU组合
            if pred['confidence'] >= self.confidence_threshold:
                if best_iou >= self.iou_threshold:
                    analysis['confidence_ranges']['high_conf_high_iou'] += 1
                else:
                    analysis['confidence_ranges']['high_conf_low_iou'] += 1
            else:
                if best_iou >= self.iou_threshold:
                    analysis['confidence_ranges']['low_conf_high_iou'] += 1
                else:
                    analysis['confidence_ranges']['low_conf_low_iou'] += 1
            
            # 分类IoU范围
            if best_iou < 0.3:
                analysis['iou_ranges']['very_low'] += 1
            elif best_iou < 0.5:
                analysis['iou_ranges']['low'] += 1
            elif best_iou < 0.7:
                analysis['iou_ranges']['medium'] += 1
            else:
                analysis['iou_ranges']['high'] += 1
        
        # 计算统计信息
        if ious:
            analysis['iou_stats'] = {
                'mean': np.mean(ious),
                'std': np.std(ious),
                'min': np.min(ious),
                'max': np.max(ious)
            }
        
        if confidences:
            analysis['confidence_stats'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        return analysis
    
    def visualize_difficulty_distribution(self, 
                                        predictions: List[Dict], 
                                        ground_truth: List[Dict],
                                        save_path: Optional[str] = None):
        """
        可视化困难样本分布
        
        Args:
            predictions: 预测结果
            ground_truth: 真实标注
            save_path: 保存路径（可选）
        """
        # 收集数据
        ious = []
        confidences = []
        colors = []
        
        for pred in predictions:
            best_iou = 0.0
            for gt in ground_truth:
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                best_iou = max(best_iou, iou)
            
            ious.append(best_iou)
            confidences.append(pred['confidence'])
            
            # 根据类型设置颜色
            if pred['confidence'] >= self.confidence_threshold and best_iou >= self.iou_threshold:
                colors.append('green')  # 真阳性
            elif pred['confidence'] >= self.confidence_threshold and best_iou < self.iou_threshold:
                colors.append('red')    # 假阳性
            elif pred['confidence'] < self.confidence_threshold and best_iou >= self.iou_threshold:
                colors.append('orange') # 假阴性
            else:
                colors.append('blue')   # 真阴性
        
        # 创建图形
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 散点图：置信度 vs IoU
        scatter = ax1.scatter(confidences, ious, c=colors, alpha=0.6)
        ax1.axhline(y=self.iou_threshold, color='black', linestyle='--', alpha=0.5)
        ax1.axvline(x=self.confidence_threshold, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('IoU')
        ax1.set_title('Confidence vs IoU Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 置信度分布直方图
        ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=self.confidence_threshold, color='red', linestyle='--', label='Threshold')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # IoU分布直方图
        ax3.hist(ious, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(x=self.iou_threshold, color='red', linestyle='--', label='Threshold')
        ax3.set_xlabel('IoU')
        ax3.set_ylabel('Count')
        ax3.set_title('IoU Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 困难样本分布图已保存: {save_path}")
        
        plt.show()
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        """
        生成硬负样本挖掘报告
        
        Args:
            output_path: 报告保存路径（可选）
            
        Returns:
            报告字典
        """
        # 计算准确率指标
        total = self.stats['total_predictions']
        if total > 0:
            precision = self.stats['true_positives'] / (self.stats['true_positives'] + self.stats['false_positives']) if (self.stats['true_positives'] + self.stats['false_positives']) > 0 else 0
            recall = self.stats['true_positives'] / (self.stats['true_positives'] + self.stats['false_negatives']) if (self.stats['true_positives'] + self.stats['false_negatives']) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1_score = 0
        
        # 计算统计信息
        confidence_stats = {}
        iou_stats = {}
        
        if self.stats['confidence_distribution']:
            confidences = np.array(self.stats['confidence_distribution'])
            confidence_stats = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences))
            }
        
        if self.stats['iou_distribution']:
            ious = np.array(self.stats['iou_distribution'])
            iou_stats = {
                'mean': float(np.mean(ious)),
                'std': float(np.std(ious)),
                'min': float(np.min(ious)),
                'max': float(np.max(ious)),
                'median': float(np.median(ious))
            }
        
        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold,
                'hard_negative_ratio': self.hard_negative_ratio
            },
            'statistics': {
                'total_predictions': self.stats['total_predictions'],
                'hard_negatives': self.stats['hard_negatives'],
                'false_positives': self.stats['false_positives'],
                'false_negatives': self.stats['false_negatives'],
                'true_positives': self.stats['true_positives']
            },
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'hard_negative_rate': self.stats['hard_negatives'] / total if total > 0 else 0
            },
            'distributions': {
                'confidence': confidence_stats,
                'iou': iou_stats
            }
        }
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📊 硬负样本挖掘报告已保存: {output_path}")
        
        return report
    
    def print_summary(self):
        """打印挖掘摘要"""
        print("\n" + "="*60)
        print("📊 硬负样本挖掘摘要")
        print("="*60)
        print(f"总预测数: {self.stats['total_predictions']}")
        print(f"硬负样本数: {self.stats['hard_negatives']}")
        print(f"假阳性数: {self.stats['false_positives']}")
        print(f"假阴性数: {self.stats['false_negatives']}")
        print(f"真阳性数: {self.stats['true_positives']}")
        
        if self.stats['total_predictions'] > 0:
            hard_negative_rate = self.stats['hard_negatives'] / self.stats['total_predictions']
            print(f"硬负样本率: {hard_negative_rate:.2%}")
        
        if self.stats['confidence_distribution']:
            confidences = np.array(self.stats['confidence_distribution'])
            print(f"平均置信度: {np.mean(confidences):.3f}")
        
        if self.stats['iou_distribution']:
            ious = np.array(self.stats['iou_distribution'])
            print(f"平均IoU: {np.mean(ious):.3f}")
        
        print("="*60)


def integrate_with_yolo_training():
    """
    展示如何将硬负样本挖掘集成到YOLO训练中
    
    这是一个示例函数，展示如何在训练过程中使用硬负样本挖掘
    """
    print("🔧 硬负样本挖掘集成示例")
    print("="*50)
    
    # 创建挖掘器
    miner = HardNegativeMiner(
        confidence_threshold=0.5,
        iou_threshold=0.5,
        hard_negative_ratio=0.3,
        save_samples=True
    )
    
    # 模拟预测结果
    predictions = [
        {'bbox': [100, 100, 200, 200], 'confidence': 0.8, 'class': 0},
        {'bbox': [300, 300, 400, 400], 'confidence': 0.3, 'class': 0},
        {'bbox': [500, 500, 600, 600], 'confidence': 0.9, 'class': 0}
    ]
    
    # 模拟真实标注
    ground_truth = [
        {'bbox': [110, 110, 210, 210], 'class': 0},
        {'bbox': [520, 520, 620, 620], 'class': 0}
    ]
    
    # 进行硬负样本挖掘
    hard_negatives = miner.find_hard_negatives(predictions, ground_truth)
    
    print(f"找到 {len(hard_negatives)} 个硬负样本")
    for i, hn in enumerate(hard_negatives):
        print(f"  {i+1}. 类型: {hn['type']}, 置信度: {hn['confidence']:.3f}, IoU: {hn['iou']:.3f}")
    
    # 分析困难样本分布
    analysis = miner.analyze_difficulty_distribution(predictions, ground_truth)
    print(f"\n困难样本分析:")
    print(f"  高置信度低IoU: {analysis['confidence_ranges']['high_conf_low_iou']}")
    print(f"  高置信度高IoU: {analysis['confidence_ranges']['high_conf_high_iou']}")
    print(f"  低置信度低IoU: {analysis['confidence_ranges']['low_conf_low_iou']}")
    print(f"  低置信度高IoU: {analysis['confidence_ranges']['low_conf_high_iou']}")
    
    # 生成报告
    report = miner.generate_report()
    print(f"\n模型性能指标:")
    print(f"  精确率: {report['metrics']['precision']:.3f}")
    print(f"  召回率: {report['metrics']['recall']:.3f}")
    print(f"  F1分数: {report['metrics']['f1_score']:.3f}")
    
    # 打印摘要
    miner.print_summary()


if __name__ == "__main__":
    # 运行集成示例
    integrate_with_yolo_training()
