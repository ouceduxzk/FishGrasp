#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增量训练脚本 - 每500个图片训练一次
支持自动收集新数据、批处理训练、模型更新等功能

功能特性：
1. 监控指定目录，收集新的标注数据
2. 每积累500个图片自动触发训练
3. 支持从上次训练的模型继续训练
4. 自动管理训练历史和模型版本
5. 支持训练状态恢复和错误处理

使用示例：
  # 基础使用
  python3 detection/incremental_train.py \
    --data_dir ./datasets/incremental \
    --batch_size 500 \
    --model_path runs/train/yolov8s_strong_20250909_114704/weights/best.pt

  # 自定义配置
  python3 detection/incremental_train.py \
    --data_dir ./datasets/incremental \
    --batch_size 500 \
    --model_path runs/train/yolov8s_strong_20250909_114704/weights/best.pt \
    --epochs 50 \
    --imgsz 640 \
    --device 0 \
    --project runs/incremental_train \
    --check_interval 300
"""

import os
import sys
import json
import time
import shutil
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('incremental_train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IncrementalTrainer:
    """增量训练器"""
    
    def __init__(self, args):
        self.args = args
        self.data_dir = Path(args.data_dir)
        self.batch_size = args.batch_size
        self.model_path = args.model_path
        self.epochs = args.epochs
        self.imgsz = args.imgsz
        self.device = args.device
        self.project = args.project
        self.check_interval = args.check_interval
        
        # 创建必要目录
        self.project_dir = Path(self.project)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # 状态文件路径
        self.state_file = self.project_dir / "training_state.json"
        self.processed_file = self.project_dir / "processed_files.txt"
        
        # 训练状态
        self.training_state = self.load_training_state()
        self.processed_files = self.load_processed_files()
        
        # 当前批次数据
        self.current_batch = []
        self.batch_counter = 0
        
        logger.info(f"增量训练器初始化完成")
        logger.info(f"数据目录: {self.data_dir}")
        logger.info(f"批次大小: {self.batch_size}")
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"检查间隔: {self.check_interval}秒")
    
    def load_training_state(self) -> Dict:
        """加载训练状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logger.info(f"加载训练状态: {state}")
                return state
            except Exception as e:
                logger.warning(f"加载训练状态失败: {e}")
        
        # 默认状态
        return {
            "last_training_time": None,
            "total_batches": 0,
            "total_images": 0,
            "current_model": self.model_path,
            "training_history": []
        }
    
    def save_training_state(self):
        """保存训练状态"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_state, f, ensure_ascii=False, indent=2)
            logger.info("训练状态已保存")
        except Exception as e:
            logger.error(f"保存训练状态失败: {e}")
    
    def load_processed_files(self) -> set:
        """加载已处理文件列表"""
        if self.processed_file.exists():
            try:
                with open(self.processed_file, 'r', encoding='utf-8') as f:
                    files = set(line.strip() for line in f if line.strip())
                logger.info(f"加载已处理文件: {len(files)} 个")
                return files
            except Exception as e:
                logger.warning(f"加载已处理文件失败: {e}")
        return set()
    
    def save_processed_files(self):
        """保存已处理文件列表"""
        try:
            with open(self.processed_file, 'w', encoding='utf-8') as f:
                for file_path in sorted(self.processed_files):
                    f.write(f"{file_path}\n")
            logger.info(f"已处理文件列表已保存: {len(self.processed_files)} 个")
        except Exception as e:
            logger.error(f"保存已处理文件列表失败: {e}")
    
    def collect_new_data(self) -> List[Path]:
        """收集新的数据文件"""
        if not self.data_dir.exists():
            logger.warning(f"数据目录不存在: {self.data_dir}")
            return []
        
        # 支持的图像格式
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        # 收集所有图像文件
        all_images = []
        for ext in image_exts:
            all_images.extend(self.data_dir.rglob(f"*{ext}"))
            all_images.extend(self.data_dir.rglob(f"*{ext.upper()}"))
        
        # 过滤出未处理的文件
        new_images = []
        for img_path in all_images:
            img_str = str(img_path)
            if img_str not in self.processed_files:
                new_images.append(img_path)
        
        logger.info(f"发现 {len(new_images)} 个新图像文件")
        return new_images
    
    def create_batch_dataset(self, image_files: List[Path]) -> Optional[Path]:
        """创建批次数据集"""
        if not image_files:
            return None
        
        # 创建批次目录
        batch_dir = self.project_dir / f"batch_{self.batch_counter:04d}"
        batch_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        images_dir = batch_dir / "images"
        labels_dir = batch_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # 复制图像文件
        copied_images = []
        for img_path in image_files:
            try:
                # 复制图像
                dst_img = images_dir / img_path.name
                shutil.copy2(img_path, dst_img)
                copied_images.append(dst_img)
                
                # 查找对应的标签文件
                label_path = img_path.parent / f"{img_path.stem}.txt"
                if label_path.exists():
                    dst_label = labels_dir / f"{img_path.stem}.txt"
                    shutil.copy2(label_path, dst_label)
                else:
                    logger.warning(f"未找到标签文件: {label_path}")
                
            except Exception as e:
                logger.error(f"复制文件失败 {img_path}: {e}")
        
        # 创建dataset.yaml
        dataset_yaml = {
            'path': str(batch_dir.absolute()),
            'train': 'images',
            'val': 'images',  # 使用相同数据作为验证集
            'test': '',
            'nc': 1,  # 类别数量
            'names': ['fish']  # 类别名称
        }
        
        yaml_path = batch_dir / "dataset.yaml"
        import yaml
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        logger.info(f"批次数据集创建完成: {batch_dir}")
        logger.info(f"包含 {len(copied_images)} 个图像")
        
        return batch_dir
    
    def train_model(self, dataset_path: Path) -> Optional[Path]:
        """训练模型"""
        try:
            from ultralytics import YOLO
            
            # 加载模型
            model = YOLO(self.training_state["current_model"])
            
            # 训练参数
            train_args = {
                'data': str(dataset_path / "dataset.yaml"),
                'epochs': self.epochs,
                'imgsz': self.imgsz,
                'device': self.device if self.device else None,
                'project': str(self.project_dir),
                'name': f"batch_{self.batch_counter:04d}",
                'save': True,
                'save_period': 10,  # 每10个epoch保存一次
                'verbose': True,
                'patience': 20,  # 早停耐心值
                'lr0': 0.01,  # 初始学习率
                'lrf': 0.01,  # 最终学习率
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'plots': True,
                'source': None,
                'rect': False,
                'cos_lr': False,
                'close_mosaic': 10,
                'resume': False,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'multi_scale': False,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'plots': True,
                'source': None,
                'rect': False,
                'cos_lr': False,
                'close_mosaic': 10,
                'resume': False,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'multi_scale': False,
            }
            
            logger.info(f"开始训练批次 {self.batch_counter}")
            logger.info(f"数据集路径: {dataset_path}")
            logger.info(f"训练参数: epochs={self.epochs}, imgsz={self.imgsz}")
            
            # 开始训练
            start_time = time.time()
            results = model.train(**train_args)
            training_time = time.time() - start_time
            
            # 获取最佳模型路径
            best_model_path = self.project_dir / f"batch_{self.batch_counter:04d}" / "weights" / "best.pt"
            
            if best_model_path.exists():
                # 更新当前模型路径
                self.training_state["current_model"] = str(best_model_path)
                
                # 记录训练历史
                training_record = {
                    "batch_id": self.batch_counter,
                    "timestamp": datetime.now().isoformat(),
                    "training_time": training_time,
                    "dataset_size": len(list((dataset_path / "images").glob("*"))),
                    "model_path": str(best_model_path),
                    "metrics": {
                        "mAP50": getattr(results, 'box', {}).get('map50', 0) if hasattr(results, 'box') else 0,
                        "mAP50-95": getattr(results, 'box', {}).get('map', 0) if hasattr(results, 'box') else 0,
                    } if results else {}
                }
                
                self.training_state["training_history"].append(training_record)
                self.training_state["total_batches"] += 1
                self.training_state["total_images"] += training_record["dataset_size"]
                self.training_state["last_training_time"] = training_record["timestamp"]
                
                logger.info(f"训练完成: {training_time:.2f}秒")
                logger.info(f"最佳模型: {best_model_path}")
                logger.info(f"训练指标: {training_record['metrics']}")
                
                return best_model_path
            else:
                logger.error(f"训练完成但未找到最佳模型: {best_model_path}")
                return None
                
        except Exception as e:
            logger.error(f"训练失败: {e}")
            return None
    
    def process_batch(self, image_files: List[Path]):
        """处理一个批次的数据"""
        if len(image_files) < self.batch_size:
            logger.info(f"图像数量不足 ({len(image_files)} < {self.batch_size})，等待更多数据")
            return
        
        # 取前batch_size个文件
        batch_files = image_files[:self.batch_size]
        
        logger.info(f"开始处理批次 {self.batch_counter}，包含 {len(batch_files)} 个图像")
        
        # 创建数据集
        dataset_path = self.create_batch_dataset(batch_files)
        if not dataset_path:
            logger.error("创建数据集失败")
            return
        
        # 训练模型
        model_path = self.train_model(dataset_path)
        if model_path:
            # 标记文件为已处理
            for img_path in batch_files:
                self.processed_files.add(str(img_path))
            
            # 保存状态
            self.save_training_state()
            self.save_processed_files()
            
            logger.info(f"批次 {self.batch_counter} 处理完成")
            self.batch_counter += 1
        else:
            logger.error(f"批次 {self.batch_counter} 训练失败")
    
    def run(self):
        """运行增量训练"""
        logger.info("开始增量训练监控")
        
        try:
            while True:
                # 收集新数据
                new_images = self.collect_new_data()
                
                if new_images:
                    logger.info(f"发现 {len(new_images)} 个新图像")
                    
                    # 处理批次
                    self.process_batch(new_images)
                
                # 等待下次检查
                logger.info(f"等待 {self.check_interval} 秒后再次检查...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在保存状态...")
            self.save_training_state()
            self.save_processed_files()
            logger.info("增量训练已停止")
        except Exception as e:
            logger.error(f"增量训练出错: {e}")
            self.save_training_state()
            self.save_processed_files()
            raise


def parse_args():
    """解析命令行参数"""
    ap = argparse.ArgumentParser(description="增量训练脚本 - 每500个图片训练一次")
    
    # 必需参数
    ap.add_argument("--data_dir", required=True, help="数据目录路径")
    ap.add_argument("--model_path", required=True, help="初始模型路径")
    
    # 训练参数
    ap.add_argument("--batch_size", type=int, default=500, help="每批次图像数量")
    ap.add_argument("--epochs", type=int, default=50, help="训练轮数")
    ap.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    ap.add_argument("--device", default="", help="训练设备 (如 '0' 或 'cpu')")
    
    # 系统参数
    ap.add_argument("--project", default="runs/incremental_train", help="项目输出目录")
    ap.add_argument("--check_interval", type=int, default=300, help="检查新数据间隔(秒)")
    
    return ap.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查依赖
    try:
        import ultralytics
        import yaml
        logger.info(f"Ultralytics版本: {ultralytics.__version__}")
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        logger.error("请安装: pip install ultralytics pyyaml")
        sys.exit(1)
    
    # 检查路径
    if not Path(args.model_path).exists():
        logger.error(f"模型文件不存在: {args.model_path}")
        sys.exit(1)
    
    # 创建训练器并运行
    trainer = IncrementalTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()






