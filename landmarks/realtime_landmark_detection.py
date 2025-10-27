#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时鱼体关键点检测

集成到主系统中，用于实时检测鱼的关键点并计算精确的抓取位置
"""

import cv2
import numpy as np
import torch
import time
from typing import Tuple, Optional, Dict, Any
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fish_landmark_detector import FishLandmarkDetector


class RealtimeFishLandmarkDetector:
    """实时鱼体关键点检测器"""
    
    def __init__(self, model_path: str, device: str = 'cuda', confidence_threshold: float = 0.5):
        """
        初始化实时检测器
        
        Args:
            model_path: 训练好的模型路径
            device: 计算设备 ('cuda' 或 'cpu')
            confidence_threshold: 关键点可见性阈值
        """
        self.detector = FishLandmarkDetector(model_path=model_path, device=device)
        self.confidence_threshold = confidence_threshold
        self.landmark_names = self.detector.landmark_names
        
        # 性能统计
        self.inference_times = []
        self.last_detection_time = 0
        
        print(f"✅ 鱼体关键点检测器初始化完成")
        print(f"   模型: {model_path}")
        print(f"   设备: {device}")
        print(f"   关键点: {self.landmark_names}")
    
    def detect_landmarks(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        检测图像中的鱼体关键点
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            landmarks: 关键点坐标 (N, 2) 或 None
            visibility: 关键点可见性 (N,) 或 None
            info: 检测信息字典
        """
        start_time = time.time()
        
        # 转换颜色格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            # 预测关键点
            landmarks, visibility = self.detector.predict(image_rgb)
            
            # 过滤低置信度的关键点
            valid_mask = visibility > self.confidence_threshold
            valid_landmarks = landmarks[valid_mask]
            valid_visibility = visibility[valid_mask]
            
            # 计算推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.last_detection_time = inference_time
            
            # 准备返回信息
            info = {
                'inference_time': inference_time,
                'num_valid_landmarks': np.sum(valid_mask),
                'total_landmarks': len(landmarks),
                'confidence_scores': visibility.tolist(),
                'valid_landmarks': valid_landmarks.tolist() if len(valid_landmarks) > 0 else [],
                'detection_success': len(valid_landmarks) > 0
            }
            
            if len(valid_landmarks) > 0:
                return valid_landmarks, valid_visibility, info
            else:
                return None, None, info
                
        except Exception as e:
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            info = {
                'inference_time': inference_time,
                'error': str(e),
                'detection_success': False
            }
            
            return None, None, info
    
    def calculate_grasp_point(self, landmarks: np.ndarray, visibility: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        基于关键点计算精确的抓取点
        
        Args:
            landmarks: 关键点坐标 (N, 2)
            visibility: 关键点可见性 (N,)
            
        Returns:
            grasp_point: 抓取点坐标 (2,)
            info: 计算信息
        """
        if landmarks is None or len(landmarks) == 0:
            return np.array([0, 0]), {'method': 'default', 'confidence': 0.0}
        
        # 方法1: 使用头部和身体中心的中点
        if len(landmarks) >= 2:
            # 假设前两个点是头部和身体中心
            head_center = landmarks[0]
            body_center = landmarks[1]
            grasp_point = (head_center + body_center) / 2
            method = 'head_body_midpoint'
            confidence = np.mean(visibility[:2])
        
        # 方法2: 使用所有可见点的中心
        elif len(landmarks) >= 1:
            grasp_point = np.mean(landmarks, axis=0)
            method = 'all_points_center'
            confidence = np.mean(visibility)
        
        # 方法3: 默认位置
        else:
            grasp_point = np.array([0, 0])
            method = 'default'
            confidence = 0.0
        
        info = {
            'method': method,
            'confidence': float(confidence),
            'num_landmarks_used': len(landmarks),
            'landmarks': landmarks.tolist(),
            'visibility': visibility.tolist()
        }
        
        return grasp_point, info
    
    def visualize_detection(self, image: np.ndarray, landmarks: Optional[np.ndarray], 
                           visibility: Optional[np.ndarray], grasp_point: Optional[np.ndarray] = None,
                           info: Optional[Dict] = None) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            landmarks: 检测到的关键点
            visibility: 关键点可见性
            grasp_point: 计算的抓取点
            info: 检测信息
            
        Returns:
            vis_image: 可视化图像
        """
        vis_image = image.copy()
        
        # 绘制关键点
        if landmarks is not None and len(landmarks) > 0:
            colors = [(0, 0, 255), (0, 255, 0)]  # 红色(头部), 绿色(身体)
            
            for i, (landmark, vis) in enumerate(zip(landmarks, visibility)):
                if vis > self.confidence_threshold:
                    x, y = int(landmark[0]), int(landmark[1])
                    color = colors[i % len(colors)]
                    
                    # 绘制关键点
                    cv2.circle(vis_image, (x, y), 8, color, -1)
                    cv2.circle(vis_image, (x, y), 12, (255, 255, 255), 2)
                    
                    # 添加标签
                    label = self.landmark_names[i] if i < len(self.landmark_names) else f'Point_{i}'
                    cv2.putText(vis_image, label, (x + 15, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制抓取点
        if grasp_point is not None:
            x, y = int(grasp_point[0]), int(grasp_point[1])
            cv2.circle(vis_image, (x, y), 15, (255, 0, 255), -1)  # 紫色大圆
            cv2.circle(vis_image, (x, y), 20, (255, 255, 255), 3)  # 白色边框
            cv2.putText(vis_image, 'GRASP', (x + 25, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # 添加信息文本
        if info:
            y_offset = 30
            cv2.putText(vis_image, f"FPS: {1.0/info.get('inference_time', 0.001):.1f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            if 'detection_success' in info:
                status = "SUCCESS" if info['detection_success'] else "FAILED"
                color = (0, 255, 0) if info['detection_success'] else (0, 0, 255)
                cv2.putText(vis_image, f"Detection: {status}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
            
            if 'confidence' in info:
                cv2.putText(vis_image, f"Confidence: {info['confidence']:.2f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            'avg_inference_time': float(np.mean(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'std_inference_time': float(np.std(times)),
            'avg_fps': float(1.0 / np.mean(times)),
            'total_inferences': len(times)
        }
    
    def reset_stats(self):
        """重置性能统计"""
        self.inference_times = []


def test_realtime_detection(model_path: str, camera_index: int = 0):
    """测试实时检测功能"""
    
    print("="*60)
    print("🎥 实时鱼体关键点检测测试")
    print("="*60)
    
    # 初始化检测器
    detector = RealtimeFishLandmarkDetector(model_path=model_path)
    
    # 初始化摄像头
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_index}")
        return
    
    print("✅ 摄像头初始化成功")
    print("📋 操作说明:")
    print("   - 按 'q' 退出")
    print("   - 按 'r' 重置统计")
    print("   - 按 's' 保存当前帧")
    
    frame_count = 0
    save_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头帧")
                break
            
            frame_count += 1
            
            # 检测关键点
            landmarks, visibility, info = detector.detect_landmarks(frame)
            
            # 计算抓取点
            if landmarks is not None:
                grasp_point, grasp_info = detector.calculate_grasp_point(landmarks, visibility)
                info.update(grasp_info)
            else:
                grasp_point = None
            
            # 可视化结果
            vis_frame = detector.visualize_detection(frame, landmarks, visibility, grasp_point, info)
            
            # 显示结果
            # 检查是否支持GUI显示
            try:
                cv2.imshow('Fish Landmark Detection', vis_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as e:
                if "not implemented" in str(e).lower():
                    print("⚠️  OpenCV GUI不支持，跳过图像显示")
                    key = 0  # 设置默认值
                else:
                    raise e
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset_stats()
                print("📊 统计已重置")
            elif key == ord('s'):
                save_path = f"landmark_detection_{save_count:03d}.jpg"
                cv2.imwrite(save_path, vis_frame)
                save_count += 1
                print(f"💾 图像已保存: {save_path}")
            
            # 每100帧显示一次统计
            if frame_count % 100 == 0:
                stats = detector.get_performance_stats()
                if stats:
                    print(f"📊 性能统计 (帧 {frame_count}): "
                          f"FPS={stats['avg_fps']:.1f}, "
                          f"推理时间={stats['avg_inference_time']*1000:.1f}ms")
    
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    
    finally:
        # 清理资源
        cap.release()
        # 安全地关闭所有OpenCV窗口
        try:
            cv2.destroyAllWindows()
        except cv2.error as e:
            if "not implemented" in str(e).lower():
                print("⚠️  OpenCV GUI不支持，跳过窗口清理")
            else:
                raise e
        
        # 显示最终统计
        stats = detector.get_performance_stats()
        if stats:
            print("\n📈 最终性能统计:")
            for key, value in stats.items():
                print(f"   {key}: {value}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='实时鱼体关键点检测')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--confidence', type=float, default=0.5, help='置信度阈值')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
    
    test_realtime_detection(args.model_path, args.camera)


if __name__ == "__main__":
    main()



