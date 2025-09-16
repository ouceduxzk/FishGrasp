#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鱼体关键点检测器

使用MediaPipe或OpenPose风格的模型来检测鱼的关键点：
- 头部中心点
- 身体中心点  
- 尾部中心点
- 背鳍点
- 腹鳍点

这些关键点可以用于：
1. 计算鱼的精确中心位置
2. 估计鱼的姿态和方向
3. 确定最佳抓取点
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt issues
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Optional
import argparse
from datetime import datetime
import torch.nn.functional as F


class FishLandmarkDataset(Dataset):
    """鱼体关键点数据集"""
    
    def __init__(self, image_paths: List[str], landmarks: List[np.ndarray], 
                 transform=None, image_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            image_paths: 图像文件路径列表
            landmarks: 关键点坐标列表，每个元素是(N, 2)的numpy数组
            transform: 数据增强变换
            image_size: 目标图像尺寸
        """
        self.image_paths = image_paths
        self.landmarks = landmarks
        self.transform = transform
        self.image_size = image_size
        
        # 定义关键点类型（只有身体中心）
        self.landmark_names = [
            'body_center'       # 身体中心
        ]
        self.num_landmarks = len(self.landmark_names)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取关键点
        landmarks = self.landmarks[idx].copy()
        
        # 记录原始图像尺寸
        original_h, original_w = image.shape[:2]
        
        # 总是先进行填充处理
        # 计算缩放比例（保持宽高比）
        scale = min(self.image_size[0] / original_w, self.image_size[1] / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # 缩放图像
        image = cv2.resize(image, (new_w, new_h))
        
        # 缩放关键点坐标
        landmarks[:, 0] *= scale
        landmarks[:, 1] *= scale
        
        # 创建正方形画布并放置图像到左上角
        canvas = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        x_offset = 0  # 总是从左边开始
        y_offset = 0  # 总是从顶部开始
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image
        image = canvas
        
        # 关键点坐标不需要调整偏移量（因为x_offset=0, y_offset=0）
        
        # 归一化关键点坐标到[0, 1]
        h, w = image.shape[:2]
        landmarks_normalized = landmarks / np.array([w, h])
        
        
        # 手动归一化图像（不使用Albumentations）
        image_normalized = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_normalized = (image_normalized - mean) / std
        
        # 转换为tensor，确保是float32
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
        
        landmarks_tensor = torch.from_numpy(landmarks_normalized.astype(np.float32)).float()
        
        return {
            'image': image_tensor,
            'landmarks': landmarks_tensor,
            'image_path': image_path,
            'original_size': (original_h, original_w)
        }


class FishLandmarkModel(nn.Module):
    """鱼体关键点检测模型"""
    
    def __init__(self, num_landmarks: int = 1, backbone: str = 'resnet18'):
        super(FishLandmarkModel, self).__init__()
        
        self.num_landmarks = num_landmarks
        
        # 选择backbone
        if backbone == 'resnet18':
            import torchvision.models as models
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Identity()  # 移除最后的分类层
            feature_dim = 512
        elif backbone == 'efficientnet':
            import torchvision.models as models
            self.backbone = models.efficientnet_b0(pretrained=True)
            self.backbone.classifier = nn.Identity()
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 关键点回归头
        self.landmark_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_landmarks * 2)  # x, y坐标
        )
        
        # 可见性预测头（关键点是否可见）
        self.visibility_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_landmarks),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        landmarks = self.landmark_head(features)
        landmarks = landmarks.view(-1, self.num_landmarks, 2)
        
        visibility = self.visibility_head(features)
        
        return landmarks, visibility


class GaussianKernelLoss(nn.Module):
    """高斯核损失函数 - 为关键点周围的区域提供平滑的损失"""
    
    def __init__(self, sigma: float = 0.1, radius: float = 0.2, image_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            sigma: 高斯核的标准差
            radius: 考虑损失的最大半径（归一化坐标）
            image_size: 图像尺寸 (width, height)
        """
        super(GaussianKernelLoss, self).__init__()
        self.sigma = sigma
        self.radius = radius
        self.image_size = image_size
        
        # 创建坐标网格
        self.register_buffer('coord_grid', self._create_coord_grid())
    
    def _create_coord_grid(self):
        """创建坐标网格"""
        h, w = self.image_size[1], self.image_size[0]
        y_coords = torch.linspace(0, 1, h).view(-1, 1).repeat(1, w)
        x_coords = torch.linspace(0, 1, w).view(1, -1).repeat(h, 1)
        coords = torch.stack([x_coords, y_coords], dim=0)  # [2, H, W]
        return coords
    
    def forward(self, pred_landmarks: torch.Tensor, target_landmarks: torch.Tensor, 
                visibility: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_landmarks: 预测的关键点 [B, N, 2] (归一化坐标)
            target_landmarks: 目标关键点 [B, N, 2] (归一化坐标)
            visibility: 可见性 [B, N]
        Returns:
            loss: 高斯核损失
        """
        batch_size, num_landmarks, _ = pred_landmarks.shape
        total_loss = 0.0
        
        for b in range(batch_size):
            for n in range(num_landmarks):
                if visibility[b, n] < 0.5:  # 跳过不可见的关键点
                    continue
                
                target_point = target_landmarks[b, n]  # [2]
                pred_point = pred_landmarks[b, n]  # [2]
                
                # 计算到目标点的距离
                distances = torch.norm(self.coord_grid - target_point.view(2, 1, 1), dim=0)
                
                # 创建高斯权重图
                gaussian_weights = torch.exp(-(distances ** 2) / (2 * self.sigma ** 2))
                
                # 只在半径内计算损失
                mask = distances <= self.radius
                gaussian_weights = gaussian_weights * mask.float()
                
                # 计算预测点到所有网格点的距离
                pred_distances = torch.norm(self.coord_grid - pred_point.view(2, 1, 1), dim=0)
                
                # 高斯核损失：预测距离与高斯权重的加权和
                loss = torch.sum(gaussian_weights * pred_distances) / (torch.sum(gaussian_weights) + 1e-8)
                total_loss += loss
        
        return total_loss / (batch_size * num_landmarks + 1e-8)


class CombinedLoss(nn.Module):
    """组合损失函数：高斯核损失 + 传统回归损失"""
    
    def __init__(self, gaussian_weight: float = 0.7, regression_weight: float = 0.3, 
                 visibility_weight: float = 0.01, sigma: float = 0.1, radius: float = 0.2):
        super(CombinedLoss, self).__init__()
        self.gaussian_weight = gaussian_weight
        self.regression_weight = regression_weight
        self.visibility_weight = visibility_weight
        
        self.gaussian_loss = GaussianKernelLoss(sigma=sigma, radius=radius)
        self.regression_loss = nn.SmoothL1Loss()
        self.visibility_loss = nn.BCELoss()
    
    def forward(self, pred_landmarks: torch.Tensor, target_landmarks: torch.Tensor,
                pred_visibility: torch.Tensor, target_visibility: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_landmarks: 预测关键点 [B, N, 2]
            target_landmarks: 目标关键点 [B, N, 2]
            pred_visibility: 预测可见性 [B, N]
            target_visibility: 目标可见性 [B, N]
        Returns:
            Dict containing individual losses and total loss
        """
        # 高斯核损失
        gaussian_loss = self.gaussian_loss(pred_landmarks, target_landmarks, target_visibility)
        
        # 传统回归损失（只对可见点）
        visible_mask = target_visibility > 0.5
        if visible_mask.sum() > 0:
            visible_pred = pred_landmarks[visible_mask]
            visible_target = target_landmarks[visible_mask]
            regression_loss = self.regression_loss(visible_pred, visible_target)
        else:
            regression_loss = torch.tensor(0.0, device=pred_landmarks.device)
        
        # 可见性损失
        visibility_loss = self.visibility_loss(pred_visibility, target_visibility)
        
        # 组合损失
        total_loss = (self.gaussian_weight * gaussian_loss + 
                     self.regression_weight * regression_loss + 
                     self.visibility_weight * visibility_loss)
        
        return {
            'total': total_loss,
            'gaussian': gaussian_loss,
            'regression': regression_loss,
            'visibility': visibility_loss
        }


class FishLandmarkDetector:
    """鱼体关键点检测器主类"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.landmark_names = [
            'body_center'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def calculate_pixel_error(self, landmark_loss, image_size=(640, 480)):
        """
        将归一化坐标的损失转换为像素误差
        
        Args:
            landmark_loss: 归一化坐标的损失值
            image_size: 图像尺寸 (width, height)
            
        Returns:
            pixel_error: 平均像素误差
        """
        # 计算归一化坐标的RMSE
        if isinstance(landmark_loss, torch.Tensor):
            rmse_normalized = torch.sqrt(landmark_loss).item()
        else:
            rmse_normalized = np.sqrt(landmark_loss)
        
        # 转换为像素误差 (取宽高的平均值)
        avg_image_size = (image_size[0] + image_size[1]) / 2
        pixel_error = rmse_normalized * avg_image_size
        
        return pixel_error
    
    def visualize_padding(self, image_path: str, landmarks: np.ndarray, target_size: Tuple[int, int] = (256, 256)):
        """
        可视化填充效果，用于调试
        
        Args:
            image_path: 图像路径
            landmarks: 关键点坐标
            target_size: 目标尺寸
        """
        import matplotlib.pyplot as plt
        
        # 加载原始图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法加载图像: {image_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        
        # 计算填充后的图像
        scale = min(target_size[0] / original_w, target_size[1] / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # 缩放图像和关键点
        resized_image = cv2.resize(image, (new_w, new_h))
        scaled_landmarks = landmarks.copy().astype(np.float64)
        scaled_landmarks[:, 0] *= scale
        scaled_landmarks[:, 1] *= scale
        
        # 创建填充画布（图像放在左上角）
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        x_offset = 0  # 总是从左边开始
        y_offset = 0  # 总是从顶部开始
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
        
        # 关键点坐标不需要调整（因为x_offset=0, y_offset=0）
        final_landmarks = scaled_landmarks.copy()
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=50)
        axes[0].set_title(f'原始图像 {original_w}x{original_h}')
        axes[0].axis('off')
        
        # 填充后图像
        axes[1].imshow(canvas)
        axes[1].scatter(final_landmarks[:, 0], final_landmarks[:, 1], c='red', s=50)
        axes[1].set_title(f'填充后 {target_size[0]}x{target_size[1]} (scale={scale:.3f})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"原始尺寸: {original_w}x{original_h}")
        print(f"缩放比例: {scale:.3f}")
        print(f"填充偏移: x={x_offset}, y={y_offset}")
        print(f"原始关键点: {landmarks}")
        print(f"最终关键点: {final_landmarks}")
    
    def create_model(self, backbone: str = 'resnet18'):
        """创建模型"""
        self.model = FishLandmarkModel(
            num_landmarks=len(self.landmark_names),
            backbone=backbone
        ).to(self.device)
        return self.model
    
    def train_with_gaussian_loss(self, train_loader: DataLoader, val_loader: DataLoader, 
                                epochs: int = 100, lr: float = 0.001, save_dir: str = 'models'):
        """使用高斯核损失的训练方法"""
        if self.model is None:
            self.create_model()
        
        # 损失函数 - 使用组合损失（高斯核 + 回归 + 可见性）
        criterion = CombinedLoss(
            gaussian_weight=0.7,    # 高斯核损失权重
            regression_weight=0.3,  # 回归损失权重
            visibility_weight=0.01, # 可见性损失权重
            sigma=0.1,              # 高斯核标准差
            radius=0.2              # 高斯核半径
        ).to(self.device)
        
        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        # 训练历史
        train_losses = []
        val_losses = []
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_gaussian_loss = 0.0
            train_regression_loss = 0.0
            train_visibility_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                optimizer.zero_grad()
                
                pred_landmarks, pred_visibility = self.model(images)
                
                # 计算组合损失（高斯核 + 回归 + 可见性）
                visibility_target = torch.ones_like(pred_visibility)
                loss_dict = criterion(pred_landmarks, landmarks, pred_visibility, visibility_target)
                total_loss = loss_dict['total']
                
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += total_loss.item()
                train_gaussian_loss += loss_dict['gaussian'].item()
                train_regression_loss += loss_dict['regression'].item()
                train_visibility_loss += loss_dict['visibility'].item()
                
                if batch_idx % 10 == 0:
                    pixel_error = self.calculate_pixel_error(loss_dict['regression'])
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {total_loss.item():.4f}, '
                          f'Gaussian: {loss_dict["gaussian"].item():.4f}, '
                          f'Regression: {loss_dict["regression"].item():.4f}, '
                          f'Pixel Error: ~{pixel_error:.1f}px, '
                          f'Visibility: {loss_dict["visibility"].item():.4f}')
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_gaussian_loss = 0.0
            val_regression_loss = 0.0
            val_visibility_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    landmarks = batch['landmarks'].to(self.device)
                    
                    pred_landmarks, pred_visibility = self.model(images)
                    
                    visibility_target = torch.ones_like(pred_visibility)
                    loss_dict = criterion(pred_landmarks, landmarks, pred_visibility, visibility_target)
                    
                    val_loss += loss_dict['total'].item()
                    val_gaussian_loss += loss_dict['gaussian'].item()
                    val_regression_loss += loss_dict['regression'].item()
                    val_visibility_loss += loss_dict['visibility'].item()
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_gaussian = train_gaussian_loss / len(train_loader)
            avg_val_gaussian = val_gaussian_loss / len(val_loader)
            avg_train_regression = train_regression_loss / len(train_loader)
            avg_val_regression = val_regression_loss / len(val_loader)
            avg_train_visibility = train_visibility_loss / len(train_loader)
            avg_val_visibility = val_visibility_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # 计算像素误差
            train_pixel_error = self.calculate_pixel_error(avg_train_regression)
            val_pixel_error = self.calculate_pixel_error(avg_val_regression)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train - Total: {avg_train_loss:.4f}, '
                  f'Gaussian: {avg_train_gaussian:.4f}, '
                  f'Regression: {avg_train_regression:.4f}, '
                  f'Pixel Error: ~{train_pixel_error:.1f}px, '
                  f'Visibility: {avg_train_visibility:.4f}')
            print(f'  Val   - Total: {avg_val_loss:.4f}, '
                  f'Gaussian: {avg_val_gaussian:.4f}, '
                  f'Regression: {avg_val_regression:.4f}, '
                  f'Pixel Error: ~{val_pixel_error:.1f}px, '
                  f'Visibility: {avg_val_visibility:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 60)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(save_dir, 'best_fish_landmark_model_gaussian.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss,
                    'gaussian_weight': 0.7,
                    'regression_weight': 0.3,
                    'visibility_weight': 0.01,
                    'sigma': 0.1,
                    'radius': 0.2
                }, model_path)
                print(f'✅ 保存最佳模型: {model_path}')
        
        # 保存训练曲线
        self.plot_training_curves(train_losses, val_losses, save_dir)
        
        return train_losses, val_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, lr: float = 0.001, save_dir: str = 'models'):
        """训练模型"""
        if self.model is None:
            self.create_model()
        
        # 损失函数 - 使用组合损失（高斯核 + 回归 + 可见性）
        criterion = CombinedLoss(
            gaussian_weight=0.7,    # 高斯核损失权重
            regression_weight=0.3,  # 回归损失权重
            visibility_weight=0.01, # 可见性损失权重
            sigma=0.1,              # 高斯核标准差
            radius=0.2              # 高斯核半径
        ).to(self.device)
        
        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        # 训练历史
        train_losses = []
        val_losses = []
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_landmark_loss = 0.0
            train_visibility_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                optimizer.zero_grad()
                
                pred_landmarks, pred_visibility = self.model(images)
                
                # 计算关键点损失
                landmark_loss = landmark_criterion(pred_landmarks, landmarks)
                
                # 计算可见性损失（假设所有关键点都可见）
                visibility_target = torch.ones_like(pred_visibility)
                visibility_loss = visibility_criterion(pred_visibility, visibility_target)
                
                # 总损失 - 降低可见性损失权重，因为所有关键点都可见
                total_loss = landmark_loss + 0.01 * visibility_loss
                
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += total_loss.item()
                train_landmark_loss += landmark_loss.item()
                train_visibility_loss += visibility_loss.item()
                
                # 打印批次信息
                if batch_idx % 10 == 0:
                    pixel_error = self.calculate_pixel_error(landmark_loss)
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {total_loss.item():.4f}, Landmark: {landmark_loss.item():.4f}, '
                          f'Pixel Error: ~{pixel_error:.1f}px, Visibility: {visibility_loss.item():.4f}')
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_landmark_loss = 0.0
            val_visibility_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    landmarks = batch['landmarks'].to(self.device)
                    
                    pred_landmarks, pred_visibility = self.model(images)
                    
                    landmark_loss = landmark_criterion(pred_landmarks, landmarks)
                    visibility_target = torch.ones_like(pred_visibility)
                    visibility_loss = visibility_criterion(pred_visibility, visibility_target)
                    
                    total_loss = landmark_loss + 0.1 * visibility_loss
                    
                    val_loss += total_loss.item()
                    val_landmark_loss += landmark_loss.item()
                    val_visibility_loss += visibility_loss.item()
            
            # 计算平均损失
            train_loss /= len(train_loader)
            train_landmark_loss /= len(train_loader)
            train_visibility_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_landmark_loss /= len(val_loader)
            val_visibility_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 学习率调度
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            train_pixel_error = self.calculate_pixel_error(torch.tensor(train_landmark_loss))
            val_pixel_error = self.calculate_pixel_error(torch.tensor(val_landmark_loss))
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train - Total: {train_loss:.4f}, Landmark: {train_landmark_loss:.4f}, Pixel Error: ~{train_pixel_error:.1f}px, Visibility: {train_visibility_loss:.4f}')
            print(f'  Val   - Total: {val_loss:.4f}, Landmark: {val_landmark_loss:.4f}, Pixel Error: ~{val_pixel_error:.1f}px, Visibility: {val_visibility_loss:.4f}')
            print(f'  LR: {current_lr:.6f}')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(save_dir, 'best_fish_landmark_model.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'landmark_names': self.landmark_names,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, model_path)
                print(f'  ✅ 保存最佳模型: {model_path} (Val Loss: {val_loss:.4f})')
            
            # 早停检查
            if epoch > 20 and val_loss > best_val_loss * 1.1:
                print(f'  ⚠️  验证损失上升，考虑早停')
            
            print('-' * 60)
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, val_losses, save_dir)
        
        return train_losses, val_losses
    
    def train_without_validation_gaussian(self, train_loader: DataLoader, epochs: int = 100, 
                                         lr: float = 0.001, save_dir: str = 'models'):
        """无验证集的训练方法（使用高斯核损失）"""
        if self.model is None:
            self.create_model()
        
        # 损失函数 - 使用组合损失（高斯核 + 回归 + 可见性）
        criterion = CombinedLoss(
            gaussian_weight=0.7,    # 高斯核损失权重
            regression_weight=0.3,  # 回归损失权重
            visibility_weight=0.01, # 可见性损失权重
            sigma=0.1,              # 高斯核标准差
            radius=0.2              # 高斯核半径
        ).to(self.device)
        
        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        # 训练历史
        train_losses = []
        
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_gaussian_loss = 0.0
            train_regression_loss = 0.0
            train_visibility_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                optimizer.zero_grad()
                
                pred_landmarks, pred_visibility = self.model(images)
                
                # 计算组合损失（高斯核 + 回归 + 可见性）
                visibility_target = torch.ones_like(pred_visibility)
                loss_dict = criterion(pred_landmarks, landmarks, pred_visibility, visibility_target)
                total_loss = loss_dict['total']
                
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += total_loss.item()
                train_gaussian_loss += loss_dict['gaussian'].item()
                train_regression_loss += loss_dict['regression'].item()
                train_visibility_loss += loss_dict['visibility'].item()
                
                if batch_idx % 10 == 0:
                    pixel_error = self.calculate_pixel_error(loss_dict['regression'])
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {total_loss.item():.4f}, '
                          f'Gaussian: {loss_dict["gaussian"].item():.4f}, '
                          f'Regression: {loss_dict["regression"].item():.4f}, '
                          f'Pixel Error: ~{pixel_error:.1f}px, '
                          f'Visibility: {loss_dict["visibility"].item():.4f}')
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_train_gaussian = train_gaussian_loss / len(train_loader)
            avg_train_regression = train_regression_loss / len(train_loader)
            avg_train_visibility = train_visibility_loss / len(train_loader)
            
            train_losses.append(avg_train_loss)
            
            # 计算像素误差
            train_pixel_error = self.calculate_pixel_error(avg_train_regression)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train - Total: {avg_train_loss:.4f}, '
                  f'Gaussian: {avg_train_gaussian:.4f}, '
                  f'Regression: {avg_train_regression:.4f}, '
                  f'Pixel Error: ~{train_pixel_error:.1f}px, '
                  f'Visibility: {avg_train_visibility:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 60)
            
            # 学习率调度
            scheduler.step(avg_train_loss)
            
            # 保存模型
            if (epoch + 1) % 10 == 0:
                model_path = os.path.join(save_dir, f'fish_landmark_model_gaussian_epoch_{epoch+1}.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'gaussian_weight': 0.7,
                    'regression_weight': 0.3,
                    'visibility_weight': 0.01,
                    'sigma': 0.1,
                    'radius': 0.2
                }, model_path)
                print(f'✅ 保存模型: {model_path}')
        
        # 保存最终模型
        final_model_path = os.path.join(save_dir, 'final_fish_landmark_model_gaussian.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': epochs,
            'train_loss': avg_train_loss,
            'gaussian_weight': 0.7,
            'regression_weight': 0.3,
            'visibility_weight': 0.01,
            'sigma': 0.1,
            'radius': 0.2
        }, final_model_path)
        print(f'✅ 保存最终模型: {final_model_path}')
        
        return train_losses, []
    
    def train_without_validation(self, train_loader: DataLoader, epochs: int = 100, 
                                lr: float = 0.001, save_dir: str = 'models'):
        """无验证集的训练方法"""
        if self.model is None:
            self.create_model()
        
        # 损失函数 - 使用Smooth L1 Loss (Huber Loss) 对关键点更稳定
        landmark_criterion = nn.SmoothL1Loss()
        visibility_criterion = nn.BCELoss()
        
        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # 训练历史
        train_losses = []
        
        os.makedirs(save_dir, exist_ok=True)
        
        print("开始训练（无验证集）...")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_landmark_loss = 0.0
            train_visibility_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                optimizer.zero_grad()
                
                pred_landmarks, pred_visibility = self.model(images)
                
                # 计算关键点损失
                landmark_loss = landmark_criterion(pred_landmarks, landmarks)
                
                # 计算可见性损失（假设所有关键点都可见）
                visibility_target = torch.ones_like(pred_visibility)
                visibility_loss = visibility_criterion(pred_visibility, visibility_target)
                
                # 总损失 - 降低可见性损失权重，因为所有关键点都可见
                total_loss = landmark_loss + 0.01 * visibility_loss
                
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += total_loss.item()
                train_landmark_loss += landmark_loss.item()
                train_visibility_loss += visibility_loss.item()
                
                # 打印批次信息
                if batch_idx % 10 == 0:
                    pixel_error = self.calculate_pixel_error(landmark_loss)
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {total_loss.item():.4f}, Landmark: {landmark_loss.item():.4f}, '
                          f'Pixel Error: ~{pixel_error:.1f}px, Visibility: {visibility_loss.item():.4f}')
            
            # 计算平均损失
            train_loss /= len(train_loader)
            train_landmark_loss /= len(train_loader)
            train_visibility_loss /= len(train_loader)
            
            train_losses.append(train_loss)
            
            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train - Total: {train_loss:.4f}, Landmark: {train_landmark_loss:.4f}, Visibility: {train_visibility_loss:.4f}')
            print(f'  LR: {current_lr:.6f}')
            
            # 每10个epoch保存一次模型
            if (epoch + 1) % 10 == 0:
                model_path = os.path.join(save_dir, f'fish_landmark_model_epoch_{epoch+1}.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'landmark_names': self.landmark_names,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, model_path)
                print(f'  💾 保存模型: {model_path}')
            
            print('-' * 60)
        
        # 保存最终模型
        final_model_path = os.path.join(save_dir, 'final_fish_landmark_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'landmark_names': self.landmark_names,
            'epoch': epochs,
            'train_loss': train_loss,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, final_model_path)
        print(f'✅ 最终模型已保存: {final_model_path}')
        
        return train_losses, []
    
    def load_model(self, model_path: str):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"📁 加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 检查checkpoint格式
        if 'model_state_dict' not in checkpoint:
            raise ValueError(f"模型文件格式错误，缺少 'model_state_dict' 键")
        
        # 获取关键点数量
        if 'landmark_names' in checkpoint:
            num_landmarks = len(checkpoint['landmark_names'])
        else:
            # 从state_dict推断关键点数量
            num_landmarks = 1  # 默认值
            print("⚠️  警告: 无法从checkpoint获取关键点数量，使用默认值1")
        
        # 检测backbone类型
        backbone = 'resnet18'  # 默认值
        if 'backbone' in checkpoint:
            backbone = checkpoint['backbone']
            print(f"📋 从checkpoint检测到backbone: {backbone}")
        else:
            # 从state_dict推断backbone类型
            state_dict = checkpoint['model_state_dict']
            if any(key.startswith('backbone.features.') for key in state_dict.keys()):
                backbone = 'efficientnet'
                print("📋 从state_dict检测到backbone: efficientnet")
            elif any(key.startswith('backbone.conv1.') for key in state_dict.keys()):
                backbone = 'resnet18'
                print("📋 从state_dict检测到backbone: resnet18")
            else:
                print("⚠️  警告: 无法检测backbone类型，使用默认值resnet18")
        
        # 创建模型（使用检测到的backbone和当前配置的landmark数量）
        current_num_landmarks = len(self.landmark_names)
        self.model = FishLandmarkModel(
            num_landmarks=current_num_landmarks,
            backbone=backbone
        ).to(self.device)
        
        # 如果保存的模型有不同数量的landmarks，需要适配
        if num_landmarks != current_num_landmarks:
            print(f"⚠️  模型适配: 从 {num_landmarks} 个关键点适配到 {current_num_landmarks} 个关键点")
            # 加载兼容的权重
            state_dict = checkpoint['model_state_dict']
            model_state_dict = self.model.state_dict()
            
            # 只加载兼容的权重
            compatible_state_dict = {}
            for key, value in state_dict.items():
                if key in model_state_dict and model_state_dict[key].shape == value.shape:
                    compatible_state_dict[key] = value
                else:
                    print(f"跳过不兼容的权重: {key} (形状不匹配)")
            
            self.model.load_state_dict(compatible_state_dict, strict=False)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 设置landmark_names
        if 'landmark_names' in checkpoint:
            self.landmark_names = checkpoint['landmark_names']
        else:
            self.landmark_names = ['body_center']
            print("⚠️  警告: 使用默认关键点名称")
        
        print(f"✅ 模型加载成功，关键点数量: {num_landmarks}")
    
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测关键点"""
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型")
        
        original_h, original_w = image.shape[:2]
        target_size = (256, 256)
        
        # 计算缩放比例和填充偏移（与preprocess_image保持一致）
        scale = min(target_size[0] / original_w, target_size[1] / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        x_offset = 0  # 总是从左边开始
        y_offset = 0  # 总是从顶部开始
        
        # 预处理图像
        image_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            pred_landmarks, pred_visibility = self.model(image_tensor)
            
            # 模型输出的是归一化坐标 [0,1]，需要转换回原始图像坐标
            landmarks_normalized = pred_landmarks[0].cpu().numpy()  # [0,1] 范围
            
            # 转换到填充后的256x256坐标
            landmarks_padded = landmarks_normalized * np.array([target_size[0], target_size[1]])
            
            # 由于图像放在左上角，不需要减去偏移量
            # landmarks_unpadded = landmarks_padded - np.array([x_offset, y_offset])  # x_offset=0, y_offset=0
            
            # 缩放到原始图像尺寸
            landmarks_pixel = landmarks_padded / scale
            
            visibility = pred_visibility[0].cpu().numpy()
        
        return landmarks_pixel, visibility
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像 - 与训练时保持一致（保持宽高比并填充）"""
        original_h, original_w = image.shape[:2]
        target_size = (256, 256)
        
        # 计算缩放比例（保持宽高比）
        scale = min(target_size[0] / original_w, target_size[1] / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # 缩放图像
        image_resized = cv2.resize(image, (new_w, new_h))
        
        # 创建正方形画布并放置图像到左上角（不居中）
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        x_offset = 0  # 总是从左边开始
        y_offset = 0  # 总是从顶部开始
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image_resized
        
        # 归一化（使用与训练相同的ImageNet标准化）
        image_normalized = canvas.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_normalized = (image_normalized - mean) / std
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def visualize_landmarks(self, image: np.ndarray, landmarks: np.ndarray, 
                           visibility: np.ndarray, save_path: Optional[str] = None):
        """可视化关键点"""
        vis_image = image.copy()
        
        # 定义颜色（只有身体中心）
        colors = [
            (0, 255, 0)     # 身体中心 - 绿色
        ]
        
        for i, (landmark, vis) in enumerate(zip(landmarks, visibility)):
            if vis > 0.5:  # 只显示可见的关键点
                x, y = int(landmark[0]), int(landmark[1])
                color = colors[i % len(colors)]
                
                # 绘制关键点
                cv2.circle(vis_image, (x, y), 5, color, -1)
                cv2.circle(vis_image, (x, y), 8, (255, 255, 255), 2)
                
                # 添加标签
                cv2.putText(vis_image, self.landmark_names[i], 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image
    
    def calculate_fish_center(self, landmarks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """计算鱼的精确中心位置"""
        # 现在只有身体中心，直接使用它
        if len(landmarks) > 0 and visibility[0] > 0.5:
            fish_center = landmarks[0]  # 身体中心
        else:
            # 如果身体中心不可见，使用所有可见点
            valid_landmarks = [landmarks[i] for i in range(len(landmarks)) if visibility[i] > 0.5]
            if len(valid_landmarks) > 0:
                fish_center = np.mean(valid_landmarks, axis=0)
            else:
                fish_center = np.array([0, 0])  # 默认值
        
        return fish_center
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float], save_dir: str):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Fish Landmark Detection Training Curves')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'训练曲线已保存: {save_path}')


def create_data_transforms(image_size: Tuple[int, int] = (256, 256)):
    """创建数据增强变换 - 简化版本，只做归一化"""
    
    # 简化版本：不使用复杂的变换，让数据集手动处理填充
    train_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'image': 'image'})
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'image': 'image'})
    
    return train_transform, val_transform


def main():
    parser = argparse.ArgumentParser(description='鱼体关键点检测训练')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'efficientnet'])
    parser.add_argument('--save_dir', type=str, default='models', help='模型保存目录')
    
    args = parser.parse_args()
    
    # 这里需要实现数据加载逻辑
    # 假设数据格式为：images/ 目录包含图像，annotations.json 包含关键点标注
    print("请实现数据加载逻辑...")
    print("数据格式示例：")
    print("- images/ 目录：包含所有鱼体图像")
    print("- annotations.json：包含关键点标注")
    print("标注格式：")
    print('{"image_name.jpg": {"landmarks": [[x1,y1], [x2,y2], ...], "visibility": [1,1,0,1,...]}}')


if __name__ == "__main__":
    main()
