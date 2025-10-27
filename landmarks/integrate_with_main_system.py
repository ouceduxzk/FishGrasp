#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将鱼体关键点检测集成到主系统中

这个脚本展示了如何将关键点检测集成到你的实时分割和抓取系统中
"""

import sys
import os
import numpy as np
import cv2
import time
from typing import Optional, Tuple, Dict, Any

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realtime_landmark_detection import RealtimeFishLandmarkDetector


class EnhancedGraspSystem:
    """增强的抓取系统，集成关键点检测"""
    
    def __init__(self, landmark_model_path: str, device: str = 'cuda'):
        """
        初始化增强抓取系统
        
        Args:
            landmark_model_path: 关键点检测模型路径
            device: 计算设备
        """
        # 初始化关键点检测器
        self.landmark_detector = RealtimeFishLandmarkDetector(
            model_path=landmark_model_path,
            device=device
        )
        
        # 抓取策略配置
        self.grasp_strategies = {
            'landmark_based': self._grasp_based_on_landmarks,
            'fallback_to_bbox': self._grasp_based_on_bbox,
            'hybrid': self._hybrid_grasp_strategy
        }
        
        self.current_strategy = 'hybrid'
        self.landmark_confidence_threshold = 0.7
        
        print("✅ 增强抓取系统初始化完成")
        print(f"   关键点检测器: {landmark_model_path}")
        print(f"   抓取策略: {self.current_strategy}")
    
    def detect_and_calculate_grasp_point(self, rgb_image: np.ndarray, 
                                        depth_image: np.ndarray,
                                        bbox: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        检测关键点并计算抓取点
        
        Args:
            rgb_image: RGB图像
            depth_image: 深度图像
            bbox: 边界框 (可选，作为备选方案)
            
        Returns:
            grasp_point: 抓取点坐标 (2,)
            info: 详细信息
        """
        info = {
            'method': 'unknown',
            'confidence': 0.0,
            'landmark_detection_success': False,
            'bbox_available': bbox is not None,
            'processing_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # 方法1: 基于关键点的抓取
            if self.current_strategy in ['landmark_based', 'hybrid']:
                landmarks, visibility, landmark_info = self.landmark_detector.detect_landmarks(rgb_image)
                
                if landmarks is not None and len(landmarks) > 0:
                    # 计算基于关键点的抓取点
                    grasp_point, grasp_info = self.landmark_detector.calculate_grasp_point(landmarks, visibility)
                    
                    # 检查置信度
                    if grasp_info['confidence'] >= self.landmark_confidence_threshold:
                        info.update({
                            'method': 'landmark_based',
                            'confidence': grasp_info['confidence'],
                            'landmark_detection_success': True,
                            'landmarks': landmarks.tolist(),
                            'visibility': visibility.tolist(),
                            'grasp_info': grasp_info
                        })
                        
                        processing_time = time.time() - start_time
                        info['processing_time'] = processing_time
                        
                        return grasp_point, info
            
            # 方法2: 基于边界框的抓取（备选方案）
            if self.current_strategy in ['fallback_to_bbox', 'hybrid'] and bbox is not None:
                # 使用边界框中心作为抓取点
                x1, y1, x2, y2 = bbox
                grasp_point = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                
                info.update({
                    'method': 'bbox_fallback',
                    'confidence': 0.5,  # 边界框方法的固定置信度
                    'bbox': bbox.tolist(),
                    'landmark_detection_success': False
                })
                
                processing_time = time.time() - start_time
                info['processing_time'] = processing_time
                
                return grasp_point, info
            
            # 方法3: 默认抓取点
            h, w = rgb_image.shape[:2]
            grasp_point = np.array([w // 2, h // 2])  # 图像中心
            
            info.update({
                'method': 'default_center',
                'confidence': 0.1,
                'landmark_detection_success': False
            })
            
        except Exception as e:
            print(f"⚠️  抓取点计算出错: {e}")
            h, w = rgb_image.shape[:2]
            grasp_point = np.array([w // 2, h // 2])
            info['error'] = str(e)
        
        processing_time = time.time() - start_time
        info['processing_time'] = processing_time
        
        return grasp_point, info
    
    def _grasp_based_on_landmarks(self, landmarks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """基于关键点的抓取策略"""
        return self.landmark_detector.calculate_grasp_point(landmarks, visibility)[0]
    
    def _grasp_based_on_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """基于边界框的抓取策略"""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    def _hybrid_grasp_strategy(self, landmarks: Optional[np.ndarray], 
                              visibility: Optional[np.ndarray], 
                              bbox: Optional[np.ndarray]) -> np.ndarray:
        """混合抓取策略"""
        # 优先使用关键点，如果置信度不够则使用边界框
        if landmarks is not None and len(landmarks) > 0:
            grasp_point, grasp_info = self.landmark_detector.calculate_grasp_point(landmarks, visibility)
            if grasp_info['confidence'] >= self.landmark_confidence_threshold:
                return grasp_point
        
        # 备选：使用边界框
        if bbox is not None:
            return self._grasp_based_on_bbox(bbox)
        
        # 最后备选：图像中心
        return np.array([0, 0])  # 这里应该传入图像尺寸
    
    def visualize_enhanced_detection(self, rgb_image: np.ndarray, 
                                   grasp_point: np.ndarray, 
                                   info: Dict[str, Any],
                                   landmarks: Optional[np.ndarray] = None,
                                   visibility: Optional[np.ndarray] = None,
                                   bbox: Optional[np.ndarray] = None) -> np.ndarray:
        """可视化增强检测结果"""
        vis_image = rgb_image.copy()
        
        # 绘制边界框（如果存在）
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 黄色边界框
            cv2.putText(vis_image, 'Detection Box', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制关键点（如果存在）
        if landmarks is not None and len(landmarks) > 0:
            vis_image = self.landmark_detector.visualize_detection(
                vis_image, landmarks, visibility, grasp_point, info
            )
        else:
            # 只绘制抓取点
            x, y = int(grasp_point[0]), int(grasp_point[1])
            cv2.circle(vis_image, (x, y), 15, (255, 0, 255), -1)  # 紫色大圆
            cv2.circle(vis_image, (x, y), 20, (255, 255, 255), 3)  # 白色边框
            cv2.putText(vis_image, 'GRASP', (x + 25, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # 添加方法信息
        method = info.get('method', 'unknown')
        confidence = info.get('confidence', 0.0)
        
        cv2.putText(vis_image, f"Method: {method}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Confidence: {confidence:.2f}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 添加处理时间
        processing_time = info.get('processing_time', 0.0)
        cv2.putText(vis_image, f"Time: {processing_time*1000:.1f}ms", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def set_grasp_strategy(self, strategy: str):
        """设置抓取策略"""
        if strategy in self.grasp_strategies:
            self.current_strategy = strategy
            print(f"✅ 抓取策略已设置为: {strategy}")
        else:
            print(f"❌ 未知的抓取策略: {strategy}")
            print(f"可用策略: {list(self.grasp_strategies.keys())}")
    
    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.landmark_confidence_threshold = threshold
        print(f"✅ 置信度阈值已设置为: {threshold}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        landmark_stats = self.landmark_detector.get_performance_stats()
        return {
            'landmark_detector': landmark_stats,
            'grasp_strategy': self.current_strategy,
            'confidence_threshold': self.landmark_confidence_threshold
        }


def create_integration_example():
    """创建集成示例代码"""
    
    example_code = '''
# 在你的主系统中集成关键点检测的示例代码

# 1. 导入必要的模块
from landmarks.integrate_with_main_system import EnhancedGraspSystem

# 2. 初始化增强抓取系统
grasp_system = EnhancedGraspSystem(
    landmark_model_path='models/best_fish_landmark_model.pth',
    device='cuda'
)

# 3. 在你的主循环中使用
def enhanced_detect_and_grasp(self, color_image, depth_image, bbox):
    """增强的检测和抓取函数"""
    
    # 使用关键点检测计算精确抓取点
    grasp_point, info = grasp_system.detect_and_calculate_grasp_point(
        rgb_image=color_image,
        depth_image=depth_image,
        bbox=bbox
    )
    
    # 可视化结果
    vis_image = grasp_system.visualize_enhanced_detection(
        rgb_image=color_image,
        grasp_point=grasp_point,
        info=info,
        bbox=bbox
    )
    
    # 显示结果
    # 检查是否支持GUI显示
    try:
        cv2.imshow('Enhanced Detection', vis_image)
    except cv2.error as e:
        if "not implemented" in str(e).lower():
            print("⚠️  OpenCV GUI不支持，跳过图像显示")
        else:
            raise e
    
    # 打印检测信息
    print(f"抓取方法: {info['method']}")
    print(f"置信度: {info['confidence']:.2f}")
    print(f"处理时间: {info['processing_time']*1000:.1f}ms")
    
    # 如果检测成功，使用精确的抓取点
    if info['confidence'] > 0.5:
        # 将2D抓取点转换为3D点云中的抓取点
        # 这里需要结合深度信息
        return grasp_point, True
    else:
        # 使用备选方案
        return None, False

# 4. 配置抓取策略
grasp_system.set_grasp_strategy('hybrid')  # 混合策略
grasp_system.set_confidence_threshold(0.7)  # 置信度阈值

# 5. 获取性能统计
stats = grasp_system.get_performance_stats()
print(f"平均FPS: {stats['landmark_detector']['avg_fps']:.1f}")
'''
    
    return example_code


def main():
    """主函数 - 演示集成功能"""
    
    print("="*60)
    print("🔗 鱼体关键点检测集成演示")
    print("="*60)
    
    # 检查是否有模型文件
    model_path = "models/best_fish_landmark_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型或指定正确的模型路径")
        return
    
    # 创建集成示例代码
    example_code = create_integration_example()
    
    # 保存示例代码到文件
    example_file = "integration_example.py"
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print(f"✅ 集成示例代码已保存到: {example_file}")
    print("\n📋 集成步骤:")
    print("1. 训练关键点检测模型")
    print("2. 将 EnhancedGraspSystem 集成到你的主系统中")
    print("3. 替换原有的抓取点计算逻辑")
    print("4. 配置抓取策略和置信度阈值")
    print("5. 测试和调优")
    
    print(f"\n💡 查看 {example_file} 了解详细的集成代码示例")


if __name__ == "__main__":
    main()



