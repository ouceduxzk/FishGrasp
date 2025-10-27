#!/usr/bin/env python3
"""
实时人体分割和3D点云生成脚本

整合现有功能：
1. RealSense相机读取RGB和深度数据
2. SAM + Grounding DINO进行人体分割
3. 将掩码转换为3D点云

使用方法:
    python3 realtime_segmentation_3d.py --output_dir output_data --save_pointcloud

依赖:
    - 现有的seg.py, mask_to_3d.py, realsense_capture.py
    - pyrealsense2, opencv-python, numpy, torch
    - segment_anything, transformers, open3d



    add a training mechasim that grasp only on the fish with masked out other region 
"""

import argparse
import os
import sys
import time
import math
import numpy as np
import cv2
import torch
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# 导入现有模块的功能
from seg import init_models# process_image_cv2
from mask_to_3d import mask_to_3d_pointcloud, save_pointcloud, load_camera_intrinsics
from realsense_capture import setup_realsense, depth_to_pointcloud, save_pointcloud_to_file
# Landmark detector for AI-based grasp point estimation
try:
    # 优先以包形式导入
    from landmarks.fish_landmark_detector import FishLandmarkDetector
except Exception:
    # 兼容直接在工作区根目录运行
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'landmarks'))
    try:
        from fish_landmark_detector import FishLandmarkDetector
    except Exception:
        FishLandmarkDetector = None

# 追加自定义模块搜索路径（手眼标定目录）
_extra_paths = [
    "/home/ai/AI_perception/hand_eye_calibrate",
]
for _p in _extra_paths:
    try:
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
    except Exception:
        pass

# 导入鱼容器跟踪器
try:
    from FishContainerTracker import FishContainerTracker
except ImportError:
    print("[警告] 无法导入 FishContainerTracker，将跳过重量跟踪功能")
    FishContainerTracker = None

# 导入位置求解器
try:
    from PositionSolver import PositionSolver, ContainerConfig
except ImportError:
    print("[警告] 无法导入 PositionSolver，将跳过位置预测功能")
    PositionSolver = None
    ContainerConfig = None

class RealtimeSegmentation3D:
    def __init__(self, output_dir, device="cpu", save_pointcloud=True, intrinsics_file=None, hand_eye_file=None, bbox_selection="highest_confidence", debug=False, use_yolo=False, yolo_weights=None,
                 grasp_point_mode: str = "centroid", landmark_model_path: str = None, enable_weight_tracking: bool = True, max_container_weight: float = 12.5):
        """
        初始化实时分割和3D点云生成器
        
        Args:
            output_dir: 输出目录
            device: 运行设备 (cpu/cuda)
            save_pointcloud: 是否保存3D点云
            intrinsics_file: 相机内参JSON文件路径
            hand_eye_file: 手眼标定4x4齐次矩阵的.npy文件路径（相机→夹爪）
            bbox_selection: 边界框选择策略 ("smallest" 或 "largest" 或 "highest_confidence")
            debug: 是否启用调试模式（保存所有中间文件）
            use_yolo: 是否使用YOLO作为检测器
            yolo_weights: YOLO权重路径（.pt）
            grasp_point_mode: 抓取点模式 ("centroid" 或 "ai")
            landmark_model_path: AI关键点模型路径
            enable_weight_tracking: 是否启用重量跟踪
            max_container_weight: 容器最大重量（kg）
        """
        self.output_dir = output_dir
        self.device = device
        self.save_pointcloud = save_pointcloud
        self.bbox_selection = bbox_selection
        self.debug = debug
        self.use_yolo = use_yolo
        self.yolo_weights = yolo_weights
        # 抓取点模式：centroid 或 ai
        self.grasp_point_mode = grasp_point_mode
        self.landmark_model_path = landmark_model_path
        # 重量跟踪相关
        self.enable_weight_tracking = enable_weight_tracking
        self.max_container_weight = max_container_weight
        # 创建输出目录（仅在debug模式下创建）
        if self.debug:
            self.rgb_dir = os.path.join(output_dir, "rgb")
            self.depth_dir = os.path.join(output_dir, "depth")
            self.mask_dir = os.path.join(output_dir, "masks")
            self.pointcloud_dir = os.path.join(output_dir, "pointclouds")
            self.segmentation_dir = os.path.join(output_dir, "segmentation")
            self.detection_dir = os.path.join(output_dir, "detection")
            
            os.makedirs(self.rgb_dir, exist_ok=True)
            os.makedirs(self.depth_dir, exist_ok=True)
            os.makedirs(self.mask_dir, exist_ok=True)
            if save_pointcloud:
                os.makedirs(self.pointcloud_dir, exist_ok=True)
            os.makedirs(self.segmentation_dir, exist_ok=True)
            os.makedirs(self.detection_dir, exist_ok=True)
            print("调试模式已启用，将保存所有中间文件")
        else:
            print("正常模式，不保存中间文件")
        
        # 初始化模型
        print("正在初始化AI模型...")
        self.sam_predictor, self.grounding_dino_model, self.processor = init_models(device)
        
        if self.use_yolo:
            if not self.yolo_weights or not os.path.exists(self.yolo_weights):
                print(f"[警告] 已启用YOLO检测，但未找到权重: {self.yolo_weights}，将回退Grounding DINO")
                self.use_yolo = False
        
        # 初始化RealSense相机
        print("正在初始化RealSense相机...")
        self.pipeline, self.config = setup_realsense()
        if self.pipeline is None:
            raise RuntimeError("无法启动RealSense相机")
        
        # 获取相机内参和畸变系数
        self.fx, self.fy, self.cx, self.cy, self.dist, self.mtx = load_camera_intrinsics(intrinsics_file)
        print(f"使用相机内参: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        
        # 检查是否使用畸变校正
        if np.any(self.dist != 0):
            print("检测到畸变系数，将进行实时图像畸变校正")
            print(f"畸变系数: k1={self.dist[0]:.6f}, k2={self.dist[1]:.6f}, k3={self.dist[4]:.6f}")
        else:
            print("未检测到畸变系数，跳过畸变校正")
            self.mtx = None
            self.dist = None
        
        # 创建对齐对象
        import pyrealsense2 as rs
        self.align = rs.align(rs.stream.color)
        
        # 帧计数器
        self.frame_count = 0
        self.start_time = time.time()
        
        # 计时器
        self.timers = {
            'detection': [],
            'segmentation': [],
            'pointcloud_generation': [],
            'grasp_calculation': [],
            'robot_movement': [],
            'total_cycle': []
        }
        
        # 加载手眼标定矩阵（可选）
        self.hand_eye_transform = None  # 4x4 齐次矩阵，相机坐标→夹爪坐标
        if hand_eye_file is not None and os.path.exists(hand_eye_file):
            try:
                mat = np.load(hand_eye_file)
                if mat.shape == (4, 4):
                    self.hand_eye_transform = mat.astype(np.float32)
                    print("已加载手眼标定矩阵 (相机→夹爪):")
                    print(self.hand_eye_transform)
                else:
                    print(f"hand_eye_file 格式不正确，期望(4,4)，实际{mat.shape}，忽略。")
            except Exception as e:
                print(f"加载手眼标定矩阵失败: {e}")
        # 若未加载到，则使用硬编码的R、t（相机→夹爪）
        if self.hand_eye_transform is None:
            R_default = np.array([
                [-0.99791369, -0.06094636, -0.02130291],
                [ 0.06027516, -0.99770511,  0.03084494],
                [-0.02313391,  0.02949655,  0.99929714]
            ], dtype=np.float32)
            t_default = np.array([[0.04], [0.113], [-0.22081495]], dtype=np.float32)
            self.hand_eye_transform = np.eye(4, dtype=np.float32)
            self.hand_eye_transform[:3, :3] = R_default
            self.hand_eye_transform[:3, 3:4] = t_default
            print("使用硬编码手眼标定矩阵 (相机→夹爪):")
            print(self.hand_eye_transform)
        
        print("初始化完成！")

        # 初始化AI关键点检测器（可选）
        self.landmark_detector = None
        if self.grasp_point_mode == "ai":
            if FishLandmarkDetector is None:
                print("[警告] 无法导入 FishLandmarkDetector，将回退为质心模式")
                self.grasp_point_mode = "centroid"
            else:
                try:
                    if landmark_model_path is None:
                        print("[警告] 未提供 landmark_model_path，将回退为质心模式")
                        self.grasp_point_mode = "centroid"
                    else:
                        self.landmark_detector = FishLandmarkDetector(model_path=landmark_model_path, device=('cuda' if self.device=='cuda' and torch.cuda.is_available() else 'cpu'))
                        print(f"已加载AI关键点模型: {landmark_model_path}")
                except Exception as e:
                    print(f"[警告] 关键点模型初始化失败: {e}，将回退为质心模式")
                    self.grasp_point_mode = "centroid"

        # 初始化鱼容器跟踪器（可选）
        self.fish_tracker = None
        if self.enable_weight_tracking and FishContainerTracker is not None:
            try:
                self.fish_tracker = FishContainerTracker(
                    max_weight_kg=self.max_container_weight,
                    data_file=os.path.join(self.output_dir, "fish_tracking_data.json")
                )
                print(f"已启用鱼容器跟踪器，最大容量: {self.max_container_weight}kg")
            except Exception as e:
                print(f"[警告] 鱼容器跟踪器初始化失败: {e}")
                self.fish_tracker = None
        else:
            print("鱼容器跟踪器未启用")

        # 初始化位置求解器（可选）
        self.position_solver = None
        if self.enable_weight_tracking and PositionSolver is not None:
            try:
                # 配置容器参数（根据实际容器尺寸调整）
                container_config = ContainerConfig(
                    width_mm=300.0,      # 容器宽度
                    height_mm=200.0,     # 容器高度
                    depth_mm=150.0,      # 容器深度
                    grid_spacing_mm=30.0, # 网格间距
                    margin_mm=20.0,      # 边距
                    base_height_mm=0.0   # 基础高度
                )
                self.position_solver = PositionSolver(container_config)
                print("已启用位置求解器")
            except Exception as e:
                print(f"[警告] 位置求解器初始化失败: {e}")
                self.position_solver = None
        else:
            print("位置求解器未启用")


        import jkrc 
        self.robot = jkrc.RC("192.168.80.116")
        self.robot.login()   
        self.robot.set_digital_output(0, 0, 0)

    
    def time_step(self, step_name):
        """计时器装饰器，用于测量各个步骤的时间"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed = end_time - start_time
                self.timers[step_name].append(elapsed)
                print(f"⏱️  {step_name}: {elapsed:.3f}s")
                return result
            return wrapper
        return decorator
    
    def print_timing_summary(self):
        """打印时间统计摘要"""
        print("\n" + "="*60)
        print("📊 时间统计摘要")
        print("="*60)
        
        for step_name, times in self.timers.items():
            if times:
                avg_time = np.mean(times)
                min_time = np.min(times)
                max_time = np.max(times)
                total_time = np.sum(times)
                print(f"{step_name:20s}: 平均={avg_time:.3f}s, 最小={min_time:.3f}s, 最大={max_time:.3f}s, 总计={total_time:.3f}s")
            else:
                print(f"{step_name:20s}: 无数据")
        
        print("="*60)
    
    def capture_frames(self):
        """
        捕获RGB和深度帧
        
        Returns:
            color_image: RGB图像
            depth_image: 深度图像 (毫米)
            success: 是否成功
        """
        try:
            # 等待新的帧
            frames = self.pipeline.wait_for_frames()
            
            # 对齐深度帧到RGB帧
            aligned_frames = self.align.process(frames)
            
            # 获取对齐后的帧
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None, False
            
            # 转换为numpy数组（RealSense配置为bgr8，因此这里直接得到BGR格式，适用于OpenCV）
            color_image = np.asanyarray(color_frame.get_data())

            # 获取深度数据
            height, width = depth_frame.get_height(), depth_frame.get_width()
            depth_image = np.zeros((height, width), dtype=np.uint16)
            
            for y in range(height):
                for x in range(width):
                    dist = depth_frame.get_distance(x, y)
                    if dist > 0:
                        depth_image[y, x] = int(dist * 1000)  # 转换为毫米
            
            # 如果启用了畸变校正，校正图像
            # if self.mtx is not None and self.dist is not None:
            #     color_image = cv2.undistort(color_image, self.mtx, self.dist)
            #     # # 深度图像需要转换为float32类型进行畸变校正
            #     # depth_image_float = depth_image.astype(np.float32)
            #     # depth_image_undistorted = cv2.undistort(depth_image_float, self.mtx, self.dist)
            #     # depth_image = depth_image_undistorted.astype(np.uint16)
            
            return color_image, depth_image, True
            
        except Exception as e:
            print(f"捕获帧时出错: {e}")
            return None, None, False
    
    # write seperate function to do detection and segmentation togther but segmentation is done for all the objects, 
    # and then we can select the  grasp object based on the distance of camera,.
    def detect_and_segment_and_dump_all(self, color_image, depth_image):
        """
        检测所有候选鱼，分别分割并计算点云质心深度，选取离相机最近的一个。

        Args:
            color_image: BGR 图像 (H,W,3)
            depth_image: 深度图 (毫米, uint16)

        Returns:
            mask_np: 选中鱼的单通道uint8掩码（0/255），若失败返回None
            base_name: 文件基名字符串，若失败返回None
        """
        # 1) 检测所有候选框
        detection_start = time.time()
        if getattr(self, 'use_yolo', False):
            # YOLO 路径：detect_yolo 已返回所有满足条件的框 (x1,y1,x2,y2,conf)
            boxes = self.detect_yolo(color_image, self.yolo_weights, conf=0.25, iou=0.45, imgsz=640)
        else:
            # GroundingDINO 路径：复用 _detect_boxes 的实现逻辑但收集全部有效框
            image_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            text_prompt = "fish. crab. marine animal"
            inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.grounding_dino_model(**inputs)
            H, W = color_image.shape[0], color_image.shape[1]
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                text_threshold=0.3,
                target_sizes=[image_pil.size[::-1]]
            )
            result = results[0]
            boxes = []
            if len(result.get("boxes", [])) > 0:
                for box in result["boxes"]:
                    x1, y1, x2, y2 = [int(c) for c in box.tolist()]
                    x1 = max(0, min(x1, W - 1))
                    y1 = max(0, min(y1, H - 1))
                    x2 = max(0, min(x2, W - 1))
                    y2 = max(0, min(y2, H - 1))
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    if area > 1000:
                        # 为了统一，与 YOLO 一样附上一个伪置信度 1.0
                        boxes.append((x1, y1, x2, y2, 1.0))
        detection_time = time.time() - detection_start
        self.timers['detection'].append(detection_time)
        print(f"⏱️  detection(all): {detection_time:.3f}s  候选数: {len(boxes) if boxes else 0}")

        if not boxes:
            print("未检测到目标，跳过分割。")
            return None, None

        # 2) 逐个候选框进行分割，并计算点云质心深度
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = f"frame_{self.frame_count:06d}_{timestamp}"

        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)

        best_idx = -1
        best_depth_m = float('inf')
        best_mask = None

        segmentation_start = time.time()
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b[:4]
            boxes_tensor = torch.tensor([[x1, y1, x2, y2]], device=self.device)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_tensor, image_rgb.shape[:2])

            try:
                masks, scores, logits = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False
                )
            except Exception as e:
                print(f"[分割] 候选框 {i} 预测失败: {e}")
                continue

            if masks.shape[0] == 0 or masks.shape[1] == 0:
                print(f"[分割] 候选框 {i} 未生成掩码")
                continue

            m_bool = masks[0][0].detach().cpu().numpy().astype(np.uint8)
            mask_np = m_bool * 255
            # 限制在 bbox 内
            restricted_mask = np.zeros_like(mask_np, dtype=np.uint8)
            restricted_mask[y1:y2, x1:x2] = mask_np[y1:y2, x1:x2]
            mask_np = restricted_mask

            # 计算点云并求质心深度（相机坐标系，单位米）
            mask_bool = (mask_np > 0)
            if not np.any(mask_bool):
                print(f"[分割] 候选框 {i} 掩码为空，跳过")
                continue

            points, colors = self.generate_pointcloud(color_image, depth_image, mask_bool)
            if points is None or len(points) == 0:
                print(f"[点云] 候选框 {i} 点云为空，跳过")
                continue

            centroid = np.mean(points, axis=0)  # (x,y,z) in meters (cam frame)
            depth_m = float(centroid[2])
            print(f"候选框 {i} 质心深度: {depth_m:.4f} m  bbox=({x1},{y1},{x2},{y2})")

            # 记录调试输出
            if self.debug:
                cv2.imwrite(os.path.join(self.segmentation_dir, f"{base_name}_cand{i}_mask.png"), mask_np)
                det_vis = color_image.copy()
                cv2.rectangle(det_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(det_vis, f"cand {i}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.imwrite(os.path.join(self.detection_dir, f"{base_name}_cand{i}_box.png"), det_vis)

            if depth_m < best_depth_m:
                best_depth_m = depth_m
                best_mask = mask_np
                best_idx = i

        segmentation_time = time.time() - segmentation_start
        self.timers['segmentation'].append(segmentation_time)
        print(f"⏱️  segmentation(all): {segmentation_time:.3f}s")

        if best_idx == -1 or best_mask is None:
            print("分割/点云均失败，未选出候选")
            return None, None

        print(f"选择最近候选: idx={best_idx}, 深度={best_depth_m:.4f} m")

        if best_depth_m > 0.8:
            print(f"深度超过0.8m，跳过")
            return None, None

        # 可视化最终选择
        if self.debug:
            colored = np.zeros_like(color_image)
            colored[best_mask > 0] = [0, 255, 0]
            vis = cv2.addWeighted(color_image, 1.0, colored, 0.4, 0)
            vis_path = os.path.join(self.segmentation_dir, f"{base_name}_closest_overlay.png")
            cv2.imwrite(vis_path, vis)

        return best_mask, base_name



    def detect_and_segment_and_dump(self, color_image):
        """
        本地完成检测->分割 返回用于显示的单通道uint8掩码（0/255）。
        based on confidence score,只选择一条鱼进行分割，无检测时返回None。
        """
        # 检测（只选择一条鱼）
        detection_start = time.time()
        if getattr(self, 'use_yolo', False):
            boxes = self.detect_yolo(color_image, self.yolo_weights, conf=0.25, iou=0.45, imgsz=640)
        else:
            boxes = self._detect_boxes(color_image)
        detection_time = time.time() - detection_start
        self.timers['detection'].append(detection_time)
        print(f"⏱️  detection: {detection_time:.3f}s")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = f"frame_{self.frame_count:06d}_{timestamp}"
        
        # 保存检测可视化（仅在debug模式下）
        if self.debug:
            det_vis = color_image.copy()
            if len(boxes) > 0:
                # 只标记选中的鱼（绿色框）
                x1, y1, x2, y2, confidence = boxes[0]
                cv2.rectangle(det_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 绿色粗框表示选中的鱼
                cv2.putText(det_vis, f"SELECTED (conf: {confidence:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 保存选中的鱼的裁剪图像
                crop = color_image[y1:y2, x1:x2]
                if crop.size > 0:
                    cv2.imwrite(os.path.join(self.detection_dir, f"{base_name}_selected_fish.png"), crop)
                
                cv2.imwrite(os.path.join(self.detection_dir, f"{base_name}_dino_detection.png"), det_vis)

        if not boxes:
            print("未检测到目标，跳过分割。")
            return None, None

        # 分割（SAM）- 只处理选中的一条鱼
        segmentation_start = time.time()
        try:
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image_rgb)
            
            # 只使用选中的边界框
            x1, y1, x2, y2, confidence = boxes[0]
            boxes_tensor = torch.tensor([[x1, y1, x2, y2]], device=self.device)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_tensor, image_rgb.shape[:2])

            # 使用multimask_output=False确保只返回一个掩码
            masks, scores, logits = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False  # 只返回一个最佳掩码
            )

            # 处理选中的鱼的掩码 - 确保只使用第一个掩码
            if masks.shape[0] > 0 and masks.shape[1] > 0:
                # 只取第一个掩码（索引[0][0]）
                m_bool = masks[0][0].cpu().numpy().astype(np.uint8)
                mask_np = m_bool * 255
                
                # 进一步限制掩码只在边界框区域内
                # 创建一个全零掩码
                restricted_mask = np.zeros_like(mask_np)
                # 只在边界框区域内应用掩码
                restricted_mask[y1:y2, x1:x2] = mask_np[y1:y2, x1:x2]
                mask_np = restricted_mask
                
                # 保存掩码（仅在debug模式下）
                if self.debug:
                    mask_path = os.path.join(self.segmentation_dir, f"{base_name}_selected_fish_mask.png")
                    cv2.imwrite(mask_path, mask_np)
                    
                    # 保存裁剪掩码
                    mask_crop = mask_np[y1:y2, x1:x2]
                    if mask_crop.size > 0:
                        mask_crop_path = os.path.join(self.segmentation_dir, f"{base_name}_selected_fish_mask_crop.png")
                        cv2.imwrite(mask_crop_path, mask_crop)
                    
                    # 保存裁剪可视化
                    crop = color_image[y1:y2, x1:x2]
                    if crop.size > 0 and mask_crop.size > 0:
                        overlay = np.zeros_like(crop)
                        overlay[mask_crop > 0] = [0, 255, 0]
                        vis_crop = cv2.addWeighted(crop, 1.0, overlay, 0.4, 0)
                        vis_crop_path = os.path.join(self.segmentation_dir, f"{base_name}_selected_fish_vis.png")
                        cv2.imwrite(vis_crop_path, vis_crop)
                    
                    # 保存整体可视化
                    colored = np.zeros_like(color_image)
                    colored[mask_np > 0] = [0, 255, 0]
                    vis = cv2.addWeighted(color_image, 1.0, colored, 0.4, 0)
                    vis_path = os.path.join(self.segmentation_dir, f"{base_name}_selected_fish_overlay.png")
                    cv2.imwrite(vis_path, vis)
                
                segmentation_time = time.time() - segmentation_start
                self.timers['segmentation'].append(segmentation_time)
                print(f"⏱️  segmentation: {segmentation_time:.3f}s")
                
                print(f"成功分割选中的鱼，掩码点数: {np.sum(mask_np > 0)}")
                print(f"掩码限制在边界框内: ({x1}, {y1}) 到 ({x2}, {y2})")
                return mask_np, base_name
            else:
                segmentation_time = time.time() - segmentation_start
                self.timers['segmentation'].append(segmentation_time)
                print(f"⏱️  segmentation: {segmentation_time:.3f}s")
                print("分割失败，未生成掩码")
                return None, None
                
        except Exception as e:
            segmentation_time = time.time() - segmentation_start
            self.timers['segmentation'].append(segmentation_time)
            print(f"⏱️  segmentation: {segmentation_time:.3f}s")
            print(f"分割时出错: {e}")
            return None, None

    def _detect_boxes(self, color_image):
        """
        使用与 seg.py 相同的方式进行检测，返回bbox列表
        只选择一条鱼进行分割和抓取
        """
        # 转换为PIL图像（与 seg.py 一致）
        image_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        text_prompt = "fish. crab. marine animal"
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_dino_model(**inputs)
        h, w = color_image.shape[0], color_image.shape[1]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=0.3,
            # 与 seg.py 相同的尺寸传入方式
            target_sizes=[image_pil.size[::-1]]
        )
        result = results[0]
        boxes = []
        print("\n检测结果详情:")
        print(f"检测到的目标数量: {len(result['boxes'])}")
        if len(result["boxes"]) == 0:
            return boxes
        
        # 过滤边界框：面积必须大于1000像素
        valid_boxes = []
        for box in result["boxes"]:
            x1, y1, x2, y2 = [int(c) for c in box.tolist()]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            # 计算边界框面积
            area = (x2 - x1) * (y2 - y1)
            if area > 1000:  # 面积过滤
                valid_boxes.append(((x1, y1, x2, y2), area))
        
        if valid_boxes:
            # 根据选择策略选择边界框
            if self.bbox_selection == "smallest":
                selected_box = min(valid_boxes, key=lambda x: x[1])
                selection_type = "面积最小的"
            elif self.bbox_selection == "largest":
                selected_box = max(valid_boxes, key=lambda x: x[1])
                selection_type = "面积最大的"
            else:
                # 默认选择最小的
                selected_box = min(valid_boxes, key=lambda x: x[1])
                selection_type = "面积最小的"
                print(f"警告: 未知的选择策略 '{self.bbox_selection}'，使用默认策略 'smallest'")
            
            boxes.append(selected_box[0])
            print(f"检测到 {len(valid_boxes)} 条鱼，选择{selection_type}进行抓取，面积: {selected_box[1]} 像素")
            print(f"选择的鱼位置: {selected_box[0]}")
        else:
            print("没有满足面积要求的边界框")
        
        return boxes

    def detect_yolo(self, color_image, yolo_weights_path, conf=0.5, iou=0.45, imgsz=640, min_area=1000):
        """
        使用Ultralytics YOLO进行鱼的检测，返回所有检测到的bbox。
        
        Args:
            color_image: OpenCV BGR图像 (H,W,3)
            yolo_weights_path: 训练好的YOLO权重 .pt 路径
            conf: 置信度阈值
            iou: NMS IOU 阈值
            imgsz: 推理输入尺寸
            min_area: 过滤最小面积（像素）
        
        Returns:
            boxes: List[Tuple[x1, y1, x2, y2, confidence]] 所有满足条件的bbox；若无则返回空列表
        """
        try:
            from ultralytics import YOLO
        except Exception as e:
            print("[错误] 未找到 ultralytics，请先: pip install ultralytics")
            print(e)
            return []

        # 加载模型（每次调用加载避免与其他依赖冲突；若频繁调用可外部缓存模型实例）
        try:
            model = YOLO(yolo_weights_path)
        except Exception as e:
            print(f"[错误] 加载YOLO权重失败: {yolo_weights_path} -> {e}")
            return []

        # YOLO支持直接传入numpy图像；确保为RGB
        #image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        try:
            results = model.predict(
                source=[color_image],
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False,
                save=False
            )
        except Exception as e:
            print(f"[错误] YOLO 推理失败: {e}")
            return []

        if not results:
            return []

        res = results[0]
        boxes_np = None
        confidences = None
        try:
            # xyxy (N,4) 和 conf (N,)
            boxes_np = res.boxes.xyxy.cpu().numpy() if hasattr(res, 'boxes') and res.boxes is not None else None
            confidences = res.boxes.conf.cpu().numpy() if hasattr(res, 'boxes') and res.boxes is not None else None
        except Exception:
            boxes_np = None
            confidences = None

        if boxes_np is None or len(boxes_np) == 0:
            return []

        # 过滤面积、裁剪到图像范围，同时保存置信度信息
        H, W = color_image.shape[0], color_image.shape[1]
        valid_boxes = []
        for i, xyxy in enumerate(boxes_np):
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy[:4].tolist()]
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W - 1))
            y2 = max(0, min(y2, H - 1))
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > min_area:
                confidence = confidences[i] if confidences is not None else 0.0
                valid_boxes.append(((x1, y1, x2, y2), area, confidence))

        boxes = []
        if valid_boxes:
            # 返回所有检测到的bbox，包含置信度信息
            for bbox_info in valid_boxes:
                bbox_coords, area, confidence = bbox_info
                # 返回格式: (x1, y1, x2, y2, confidence)
                boxes.append((*bbox_coords, confidence))
            
            print(f"[YOLO] 检测到 {len(valid_boxes)} 个候选框，全部返回用于处理")
            for i, (bbox_coords, area, confidence) in enumerate(valid_boxes):
                print(f"[YOLO] 框 {i+1}: {bbox_coords}, 置信度: {confidence:.3f}, 面积: {area} 像素")
        else:
            print("[YOLO] 没有满足面积要求的边界框")

        return boxes
    
    def detect_yolo_all(self, color_image, yolo_weights_path, conf=0.5, iou=0.45, imgsz=640):
        """
        使用Ultralytics YOLO对单帧进行推理，返回所有检测到的bbox（不做面积过滤与单框选择）。
        返回：List[Tuple[int,int,int,int,float,int]] -> (x1,y1,x2,y2,conf,cls)
        """
        try:
            from ultralytics import YOLO
        except Exception as e:
            print("[错误] 未找到 ultralytics，请先: pip install ultralytics")
            print(e)
            return []

        # 加载模型（简化为每次加载；如需优化可在外部缓存）
        try:
            model = YOLO(yolo_weights_path)
        except Exception as e:
            print(f"[错误] 加载YOLO权重失败: {yolo_weights_path} -> {e}")
            return []

        # BGR -> RGB
        #image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        try:
            results = model.predict(
                source=[color_image],
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=(0 if self.device == 'cuda' else 'cpu'),
                verbose=False,
                save=False,
            )
        except Exception as e:
            print(f"[错误] YOLO 推理失败: {e}")
            return []

        if not results:
            print("[YOLO] 无检测结果")
            return []

        res = results[0]
        if not hasattr(res, 'boxes') or res.boxes is None or res.boxes.shape[0] == 0:
            print("[YOLO] boxes 为空")
            return []

        xyxy = res.boxes.xyxy.cpu().numpy()  # (N,4)
        conf_arr = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, 'conf') else None
        cls_arr = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, 'cls') else None

        all_boxes = []
        for i, b in enumerate(xyxy):
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4].tolist()]
            conf_v = float(conf_arr[i]) if conf_arr is not None else 0.0
            cls_v = int(cls_arr[i]) if cls_arr is not None else -1
            all_boxes.append((x1, y1, x2, y2, conf_v, cls_v))

        print(f"[YOLO] 检测到 {len(all_boxes)} 个框（conf>={conf}）：前3个: {all_boxes[:3]}")
        return all_boxes

    def dump_detections(self, color_image):
        """
        将检测到的目标裁剪并保存到 detection/ 目录
        """
        boxes = self._detect_boxes(color_image)
        if not boxes:
            return 0
        base_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        saved = 0
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            crop = color_image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            filename = f"frame_{self.frame_count:06d}_{base_ts}_det_{idx}.png"
            path = os.path.join(self.detection_dir, filename)
            cv2.imwrite(path, crop)
            saved += 1
        if saved:
            print(f"已保存 {saved} 个检测裁剪到: {self.detection_dir}")
        return saved

    
    def generate_pointcloud(self, color_image, depth_image, mask):
        """
        从掩码生成3D点云
        
        Args:
            color_image: RGB图像
            depth_image: 深度图像 (毫米)
            mask: 分割掩码
            
        Returns:
            points: 3D点坐标
            colors: RGB颜色
        """
        try:
            # 转换深度图像单位为米
            depth_image_meters = depth_image.astype(np.float32) / 1000.0
            
            # 转换为RGB格式
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # 使用mask_to_3d_pointcloud函数（支持畸变校正）
            points, colors = mask_to_3d_pointcloud(
                color_image_rgb, 
                depth_image_meters, 
                mask, 
                self.fx, self.fy, self.cx, self.cy,
                self.mtx, self.dist
            )
            
            return points, colors
            
        except Exception as e:
            print(f"生成点云时出错: {e}")
            return np.array([]), np.array([])

    def apply_hand_eye_transform(self, points):
        """
        将点云从相机系转换到夹爪系，使用 self.hand_eye_transform (4x4)。
        旋转矩阵：
            [[-0.99462885  0.07149648  0.07484454]
            [-0.06962775 -0.9971997   0.02728984]
            [ 0.07658608  0.021932    0.99682173]]
            平移向量：
            [[ 0.0247092 ]
            [ 0.09912939]
            [-0.25357213]]
        """
        if self.hand_eye_transform is None or points.size == 0:
            return points
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        homo = np.hstack([points.astype(np.float32), ones])  # (N,4)
        transformed = (self.hand_eye_transform @ homo.T).T  # (N,4)
        return transformed[:, :3]

    def _rpy_to_rotation_matrix(self, rx, ry, rz):
        """
        将末端的 RPY (rx, ry, rz) 转为旋转矩阵 R (基座→末端)。
        采用常见的外旋顺序 R = Rz @ Ry @ Rx。
        """
        sx, cx = np.sin(rx), np.cos(rx)
        sy, cy = np.sin(ry), np.cos(ry)
        sz, cz = np.sin(rz), np.cos(rz)

        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx,  cx]], dtype=np.float32)
        Ry = np.array([[ cy, 0, sy],
                       [  0, 1,  0],
                       [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [ 0,   0, 1]], dtype=np.float32)

        return (Rz @ Ry @ Rx).astype(np.float32)

    def _tool_offset_to_base(self, delta_tool_xyz_mm, tcp_rpy):
        """
        将夹爪(工具)坐标系下的位移(mm)转换到基坐标系下的位移(mm)。
        delta_tool_xyz_mm: [dx, dy, dz] in tool frame
        tcp_rpy: [rx, ry, rz] in radians
        返回: [dx_base, dy_base, dz_base]
        """
        rx, ry, rz = tcp_rpy
        R_base_tool = self._rpy_to_rotation_matrix(rx, ry, rz)
        delta_tool = np.asarray(delta_tool_xyz_mm, dtype=np.float32).reshape(3, 1)
        delta_base = (R_base_tool @ delta_tool).reshape(3)
        return delta_base.tolist()

    def calculate_pointcloud_bbox(self, points):
        """
        计算点云的边界框信息，用于高度和姿态估计
        
        Args:
            points: 点云坐标 (N, 3)
            
        Returns:
            bbox_info: 字典包含中心点、尺寸、边界框等
        """
        if points.size == 0:
            return None
            
        # 计算边界框
        min_coords = np.min(points, axis=0)  # [min_x, min_y, min_z]
        max_coords = np.max(points, axis=0)  # [max_x, max_y, max_z]
        
        # 计算中心点
        center = (min_coords + max_coords) / 2.0  # [center_x, center_y, center_z]
        
        # 计算尺寸
        dimensions = max_coords - min_coords  # [width, height, depth]
        
        # 计算高度（z方向）
        height = dimensions[2]  # z方向的高度
        
        # 计算8个角点
        corners = []
        for x in [min_coords[0], max_coords[0]]:
            for y in [min_coords[1], max_coords[1]]:
                for z in [min_coords[2], max_coords[2]]:
                    corners.append([x, y, z])
        corners = np.array(corners)
        
        # 计算点云的主方向（PCA）
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca.fit(points)
            principal_axes = pca.components_  # 主方向向量
            explained_variance = pca.explained_variance_ratio_  # 解释方差比例
        except ImportError:
            print("sklearn未安装，跳过PCA姿态估计")
            principal_axes = np.eye(3)
            explained_variance = [1.0, 0.0, 0.0]
        
        bbox_info = {
            'center': center,
            'dimensions': dimensions,
            'height': height,
            'min_coords': min_coords,
            'max_coords': max_coords,
            'corners': corners,
            'principal_axes': principal_axes,
            'explained_variance': explained_variance,
            'num_points': len(points)
        }
        
        return bbox_info

    def calculate_surface_normal(self, points, method='pca'):
        """
        计算点云质心处的表面法向量
        
        Args:
            points: 点云坐标 (N, 3)
            method: 法向量计算方法 ('pca', 'plane_fitting', 'nearest_neighbors')
            
        Returns:
            normal: 法向量 (3,) 单位向量
            centroid: 质心坐标 (3,)
        """
        if points.size == 0 or len(points) < 3:
            return np.array([0, 0, 1]), np.array([0, 0, 0])
        
        centroid = np.mean(points, axis=0)
        
        if method == 'pca':
            # 使用PCA计算法向量
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                pca.fit(points)
                # 最小特征值对应的特征向量就是法向量
                normal = pca.components_[2]  # 第三个主成分（最小方差方向）
            except ImportError:
                print("sklearn未安装，使用简单平面拟合")
                return self._simple_plane_fitting(points, centroid)
        
        elif method == 'plane_fitting':
            return self._simple_plane_fitting(points, centroid)
        
        elif method == 'nearest_neighbors':
            return self._nearest_neighbors_normal(points, centroid)
        
        else:
            raise ValueError(f"未知的法向量计算方法: {method}")
        
        # 确保法向量指向正确的方向（通常指向相机方向）
        # 如果法向量的z分量为负，则翻转方向
        if normal[2] < 0:
            normal = -normal
        
        # 归一化
        normal = normal / np.linalg.norm(normal)
        
        return normal, centroid

    def _simple_plane_fitting(self, points, centroid):
        """
        使用简单平面拟合计算法向量
        """
        # 将点云中心化
        centered_points = points - centroid
        
        # 构建协方差矩阵
        cov_matrix = np.cov(centered_points.T)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 最小特征值对应的特征向量就是法向量
        normal = eigenvectors[:, 0]  # 最小特征值对应的特征向量
        
        # 确保法向量指向正确的方向
        if normal[2] < 0:
            normal = -normal
        
        # 归一化
        normal = normal / np.linalg.norm(normal)
        
        return normal, centroid

    def _nearest_neighbors_normal(self, points, centroid, k=20):
        """
        使用最近邻方法计算法向量
        """
        # 计算每个点到质心的距离
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # 找到最近的k个点
        nearest_indices = np.argsort(distances)[:k]
        nearest_points = points[nearest_indices]
        
        # 使用这些最近邻点进行平面拟合
        return self._simple_plane_fitting(nearest_points, centroid)

    def normal_to_rpy(self, normal_vector, current_rpy=None):
        """
        将法向量转换为机器人末端姿态的RPY角度
        
        Args:
            normal_vector: 法向量 (3,) 单位向量，表示期望的Z轴方向
            current_rpy: 当前RPY角度 [rx, ry, rz] (可选，用于平滑过渡)
            
        Returns:
            target_rpy: 目标RPY角度 [rx, ry, rz]
        """
        # 期望的Z轴方向（法向量）
        z_target = normal_vector / np.linalg.norm(normal_vector)
        
        # 定义参考坐标系（可以根据需要调整）
        # 这里假设X轴指向机器人前方，Y轴指向机器人左侧
        x_ref = np.array([1, 0, 0])  # 参考X轴
        y_ref = np.array([0, 1, 0])  # 参考Y轴
        
        # 计算新的坐标系
        # Z轴 = 法向量
        z_new = z_target
        
        # X轴 = 参考X轴在垂直于Z轴的平面上的投影
        x_new = x_ref - np.dot(x_ref, z_new) * z_new
        x_new = x_new / np.linalg.norm(x_new)
        
        # Y轴 = Z轴 × X轴
        y_new = np.cross(z_new, x_new)
        y_new = y_new / np.linalg.norm(y_new)
        
        # 构建旋转矩阵
        R = np.column_stack([x_new, y_new, z_new])
        
        # 将旋转矩阵转换为RPY角度
        # 使用ZYX欧拉角顺序（Roll-Pitch-Yaw）
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])  # Roll
            ry = np.arctan2(-R[2, 0], sy)      # Pitch
            rz = np.arctan2(R[1, 0], R[0, 0])  # Yaw
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])  # Roll
            ry = np.arctan2(-R[2, 0], sy)       # Pitch
            rz = 0                               # Yaw
        
        target_rpy = np.array([rx, ry, rz])
        
        # 如果提供了当前RPY，进行平滑过渡
        if current_rpy is not None:
            target_rpy = self._smooth_rpy_transition(current_rpy, target_rpy)
        
        return target_rpy

    def _smooth_rpy_transition(self, current_rpy, target_rpy, max_change=0.1):
        """
        平滑RPY角度过渡，避免突变
        
        Args:
            current_rpy: 当前RPY角度
            target_rpy: 目标RPY角度
            max_change: 单次最大变化量（弧度）
            
        Returns:
            smoothed_rpy: 平滑后的RPY角度
        """
        current_rpy = np.array(current_rpy)
        target_rpy = np.array(target_rpy)
        
        # 计算角度差
        diff = target_rpy - current_rpy
        
        # 处理角度跳跃（±π）
        for i in range(3):
            if diff[i] > np.pi:
                diff[i] -= 2 * np.pi
            elif diff[i] < -np.pi:
                diff[i] += 2 * np.pi
        
        # 限制变化量
        for i in range(3):
            if abs(diff[i]) > max_change:
                diff[i] = np.sign(diff[i]) * max_change
        
        # 计算平滑后的角度
        smoothed_rpy = current_rpy + diff
        
        return smoothed_rpy

    def estimate_fish_weight(self, points_gripper, volume_factor: float = 1.0) -> float:
        """
        根据点云估算鱼的重量
        
        Args:
            points_gripper: 夹爪坐标系中的点云 (N, 3)
            volume_factor: 体积到重量的转换因子 (kg/m³)
            
        Returns:
            weight_kg: 估算的鱼重量（千克）
        """
        if points_gripper.size == 0 or len(points_gripper) < 3:
            return 0.0
        
        # 计算点云的边界框体积
        min_coords = np.min(points_gripper, axis=0)
        max_coords = np.max(points_gripper, axis=0)
        dimensions = max_coords - min_coords
        
        # 计算体积（立方米）
        volume_m3 = np.prod(dimensions)
        
        # 应用形状因子（鱼不是完美的矩形）
        shape_factor = 0.6  # 经验值，鱼的实际体积约为边界框的60%
        effective_volume = volume_m3 * shape_factor
        
        # 估算重量（假设鱼的密度约为1000 kg/m³）
        fish_density = 1000.0  # kg/m³
        weight_kg = effective_volume * fish_density * volume_factor
        
        # 限制在合理范围内
        weight_kg = max(0.1, min(weight_kg, 2.0))  # 0.1kg 到 2.0kg
        
        return weight_kg

    def calculate_grasp_pose_with_normal(self, points_gripper, current_tcp):
        """
        计算考虑法向量的抓取姿态
        
        Args:
            points_gripper: 夹爪坐标系中的点云 (N, 3)
            current_tcp: 当前TCP位置 [x, y, z, rx, ry, rz]
            
        Returns:
            grasp_pose: 抓取姿态 [x, y, z, rx, ry, rz]
            normal_info: 法向量信息字典
        """
        if points_gripper.size == 0 or len(points_gripper) < 3:
            print("点云点数不足，无法计算法向量")
            return current_tcp, None
        
        # 计算质心和法向量
        normal, centroid = self.calculate_surface_normal(points_gripper, method='pca')
        
        print(f"质心坐标: {centroid}")
        print(f"法向量: {normal}")
        
        # 将法向量转换为RPY角度
        current_rpy = current_tcp[3:6]
        target_rpy = self.normal_to_rpy(normal, current_rpy)
        
        print(f"当前RPY: {np.degrees(current_rpy)} 度")
        print(f"目标RPY: {np.degrees(target_rpy)} 度")
        

        delta_tool_mm = [centroid[0] * 1000, centroid[1] * 1000, centroid[2] * 1000]
        delta_base_xyz = self._tool_offset_to_base(delta_tool_mm, current_tcp[3:6])

        # 构建抓取姿态
        grasp_pose = np.array([
            delta_base_xyz[0],  # 转换为毫米
            delta_base_xyz[1],
            delta_base_xyz[2] -25 , # move a bit deeper  to make sure the gripper is attached with the object
            target_rpy[0],       # 保持弧度
            target_rpy[1],
            target_rpy[2]
        ])
        
        # 法向量信息
        normal_info = {
            'centroid': centroid,
            'normal': normal,
            'current_rpy': current_rpy,
            'target_rpy': target_rpy,
            'rpy_change': target_rpy - current_rpy
        }
        
        return grasp_pose, normal_info
    
    def save_results(self, color_image, depth_image, mask, points, colors):
        """
        保存所有结果
        
        Args:
            color_image: RGB图像
            depth_image: 深度图像
            mask: 分割掩码
            points: 3D点坐标
            colors: RGB颜色
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = f"frame_{self.frame_count:06d}_{timestamp}"
        
        # 保存RGB图像
        rgb_path = os.path.join(self.rgb_dir, f"{base_name}.png")
        cv2.imwrite(rgb_path, color_image)
        
        # 保存深度图像
        depth_path = os.path.join(self.depth_dir, f"{base_name}.png")
        cv2.imwrite(depth_path, depth_image.astype(np.uint16))
        
        # 保存掩码
        if mask is not None:
            mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
            
            # 创建可视化结果
            colored_mask = np.zeros_like(color_image)
            colored_mask[mask] = [0, 255, 0]  # 绿色掩码
            alpha = 0.5
            visualization = cv2.addWeighted(color_image, 1, colored_mask, alpha, 0)
            vis_path = os.path.join(self.segmentation_dir, f"{base_name}_vis.png")
            cv2.imwrite(vis_path, visualization)
        
        # 保存点云
        if self.save_pointcloud and len(points) > 0:
            pointcloud_path = os.path.join(self.pointcloud_dir, f"{base_name}_pointcloud.ply")
            save_pointcloud_to_file(points, colors, pointcloud_path)
        
        print(f"已保存第 {self.frame_count} 帧结果")
    
    def show_preview(self, color_image, depth_image, mask, detection_vis=None, landmark_vis=None):
        """
        在一个窗口中显示2x2网格：RGB、检测、分割和关键点预测结果
        """
        # 调整图像大小
        display_size = (600, 450)
        
        # 1. RGB图像
        rgb_display = cv2.resize(color_image, display_size)
        
        # 2. 检测可视化（如果没有提供，使用RGB图像）
        if detection_vis is not None:
            detection_display = cv2.resize(detection_vis, display_size)
        else:
            detection_display = rgb_display.copy()
        
        # 3. 分割可视化
        if mask is not None:
            # 将掩码转换为彩色图像
            mask_colored = np.zeros_like(color_image)
            mask_colored[mask > 0] = [0, 255, 0]  # 绿色掩码
            # 叠加到原图上
            segmentation_vis = cv2.addWeighted(color_image, 0.7, mask_colored, 0.3, 0)
        else:
            segmentation_vis = color_image.copy()
        seg_display = cv2.resize(segmentation_vis, display_size)
        
        # 4. 关键点预测可视化（如果没有提供，使用RGB图像）
        if landmark_vis is not None:
            landmark_display = cv2.resize(landmark_vis, display_size)
        else:
            landmark_display = rgb_display.copy()
        
        # 创建2x2网格
        # 第一行：RGB | 检测
        top_row = np.hstack((rgb_display, detection_display))
        # 第二行：分割 | 关键点
        bottom_row = np.hstack((seg_display, landmark_display))
        # 垂直拼接
        combined = np.vstack((top_row, bottom_row))
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (0, 255, 0)
        
        # 第一行标签
        cv2.putText(combined, "RGB", (15, 45), font, font_scale, text_color, font_thickness)
        cv2.putText(combined, "Detection", (615, 45), font, font_scale, text_color, font_thickness)
        
        # 第二行标签
        cv2.putText(combined, "Segmentation", (15, 495), font, font_scale, text_color, font_thickness)
        cv2.putText(combined, "Landmarks", (615, 495), font, font_scale, text_color, font_thickness)
        
        # 添加帧计数
        cv2.putText(combined, f"Frame: {self.frame_count}", (10, combined.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Real-time Processing (2x2 Grid)', combined)

    
    def run_realtime(self, max_frames=None, show_preview=True):
        """
        运行实时处理
        
        Args:
            max_frames: 最大帧数 (None表示无限)
            show_preview: 是否显示预览窗口
        """
        print("开始实时处理...")
        print("按 'q' 键停止")
        if self.fish_tracker is not None:
            print("按 'r' 键重置容器")
            print("按 's' 键显示状态")
            print("按 'e' 键导出数据")
        if self.position_solver is not None:
            print("按 'p' 键显示放置状态")
            print("按 'v' 键显示放置可视化")
        

        tcp_result = self.robot.get_tcp_position()
        if isinstance(tcp_result, tuple) and len(tcp_result) == 2:
            tcp_ok, original_tcp = tcp_result
        else:
            # 如果只返回一个值，假设它是位置信息
            original_tcp = tcp_result
            tcp_ok = True

        try:
            while True:
                # 整个循环计时开始
                cycle_start = time.time()
                
                # 捕获帧
                color_image, depth_image, success = self.capture_frames()
                if not success:
                    continue
                
                self.frame_count = self.frame_count + 1
               
                if self.frame_count < 10 :
                    print(f"跳过前10帧，等待相机稳定...")
                    continue
                # 检测 + 分割 + 落盘（最近目标选择）
                mask_vis, base_name = self.detect_and_segment_and_dump_all(color_image, depth_image)
                
                # 保存RGB和深度图像（仅在debug模式下）
                if self.debug and base_name is not None:
                    # 保存RGB图像
                    rgb_path = os.path.join(self.rgb_dir, f"{base_name}.png")
                    cv2.imwrite(rgb_path, color_image)
                    
                    # 保存深度图像（原始16位）
                    depth_path = os.path.join(self.depth_dir, f"{base_name}.png")
                    cv2.imwrite(depth_path, depth_image.astype(np.uint16))
                    
                    # 保存可视化深度图像（8位彩色）
                    valid_depth = depth_image > 0
                    if valid_depth.any():
                        depth_min = depth_image[valid_depth].min()
                        depth_max = depth_image[valid_depth].max()
                        depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
                        if depth_max > depth_min:
                            depth_normalized[valid_depth] = ((depth_image[valid_depth] - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                        depth_vis_path = os.path.join(self.depth_dir, f"{base_name}_visualization.png")
                        cv2.imwrite(depth_vis_path, depth_colormap)

                # 生成检测可视化
                detection_vis = None
                if mask_vis is not None:
                    # 重新运行检测以获取边界框可视化
                    if getattr(self, 'use_yolo', False):
                        boxes = self.detect_yolo(color_image, self.yolo_weights, conf=0.25, iou=0.45, imgsz=640)
                    else:
                        boxes = self._detect_boxes(color_image)
                    
                    if boxes:
                        detection_vis = color_image.copy()
                        # 绘制所有检测框
                        for i, box in enumerate(boxes):
                            if len(box) >= 4:
                                x1, y1, x2, y2 = box[:4]
                                color = (0, 255, 0) if i == 0 else (0, 255, 255)  # 第一个框用绿色，其他用黄色
                                thickness = 3 if i == 0 else 2
                                cv2.rectangle(detection_vis, (x1, y1), (x2, y2), color, thickness)
                                if len(box) >= 5:
                                    confidence = box[4]
                                    cv2.putText(detection_vis, f"{confidence:.2f}", (x1, y1-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 生成关键点预测可视化
                landmark_vis = None
                if (self.grasp_point_mode == "ai" and self.landmark_detector is not None and 
                    mask_vis is not None):
                    try:
                        # 根据掩码计算外接矩形，得到鱼的裁剪区域
                        ys, xs = np.where(mask_vis > 0)
                        if ys.size > 0 and xs.size > 0:
                            x1, y1 = int(xs.min()), int(ys.min())
                            x2, y2 = int(xs.max())+1, int(ys.max())+1
                            crop_bgr = color_image[y1:y2, x1:x2]
                            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                            
                            # 预测关键点
                            pred_landmarks, pred_visibility = self.landmark_detector.predict(crop_rgb)
                            
                            # 可视化关键点
                            landmark_vis_crop = self.landmark_detector.visualize_landmarks(crop_rgb, pred_landmarks, pred_visibility)
                            
                            # 将裁剪区域的可视化结果放回原图
                            landmark_vis = color_image.copy()
                            landmark_vis_bgr = cv2.cvtColor(landmark_vis_crop, cv2.COLOR_RGB2BGR)
                            landmark_vis[y1:y2, x1:x2] = landmark_vis_bgr
                            
                            # 绘制边界框
                            cv2.rectangle(landmark_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    except Exception as e:
                        print(f"[可视化] 关键点预测可视化失败: {e}")
                
                # 显示预览窗口
                self.show_preview(color_image, depth_image, mask_vis, detection_vis, landmark_vis)
                
                # 确保窗口显示并处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户按 'q' 键停止")
                    break
                elif key == ord('r') and self.fish_tracker is not None:
                    print("用户按 'r' 键重置容器")
                    self.fish_tracker.reset_container(confirm=True)
                elif key == ord('s') and self.fish_tracker is not None:
                    print("用户按 's' 键显示状态")
                    self.fish_tracker.print_status()
                elif key == ord('e') and self.fish_tracker is not None:
                    print("用户按 'e' 键导出数据")
                    self.fish_tracker.export_data()
                elif key == ord('p') and self.position_solver is not None:
                    print("用户按 'p' 键显示放置状态")
                    self.position_solver.print_placement_status()
                elif key == ord('v') and self.position_solver is not None:
                    print("用户按 'v' 键显示放置可视化")
                    print(self.position_solver.visualize_placements())

                # 根据掩码生成3D点云并保存（可选应用手眼标定）
                points_gripper = None  # 初始化变量

                #import pdb; pdb.set_trace()
                if mask_vis is not None and base_name is not None:
                    # 点云生成计时
                    pointcloud_start = time.time()
                    mask_bool = (mask_vis > 0)
                    points, colors = self.generate_pointcloud(color_image, depth_image, mask_bool)
                    pointcloud_time = time.time() - pointcloud_start
                    self.timers['pointcloud_generation'].append(pointcloud_time)
                    print(f"⏱️  pointcloud_generation: {pointcloud_time:.3f}s")

                    #import pdb; pdb.set_trace()
                    if len(points) > 0:
                        # 应用手眼变换：相机→夹爪
                        points_gripper = self.apply_hand_eye_transform(points)
                        
                        # 保存点云（仅在debug模式下）
                        if self.debug:
                            # 保存相机坐标系点云
                            cam_ply = os.path.join(self.pointcloud_dir, f"{base_name}_cam_pointcloud.ply")
                            save_pointcloud_to_file(points, colors, cam_ply)
                            # 保存夹爪坐标系点云
                            grip_ply = os.path.join(self.pointcloud_dir, f"{base_name}_gripper_pointcloud.ply")
                            save_pointcloud_to_file(points_gripper, colors, grip_ply)
                
                # don't forget to transform the units, the point cloud is in meter, but robot
                # control would like to be in mm. 

                # 计算点云质心和法向量（在夹爪坐标系中）
                if points_gripper is not None and len(points_gripper) > 0:
                    # 抓取点计算计时
                    grasp_calc_start = time.time()
                    
                    # 获取当前机器人TCP位置
                    tcp_result = self.robot.get_tcp_position()
                    if isinstance(tcp_result, tuple) and len(tcp_result) == 2:
                        tcp_ok, current_tcp = tcp_result
                    else:
                        # 如果只返回一个值，假设它是位置信息
                        current_tcp = tcp_result
                        tcp_ok = True
                    print(f"当前TCP位置: {current_tcp}")
                    
                    # 检查容器是否已满
                    if self.fish_tracker is not None and self.fish_tracker.is_container_full():
                        print("📦 容器已满！停止抓取新鱼。")
                        print("按 'r' 键重置容器，按 'q' 键退出")
                        # 显示状态
                        self.fish_tracker.print_status()
                        continue
                    
                    # 计算抓取点（优先AI）
                    relative_move = None
                    angle_rad = 0
                    if self.grasp_point_mode == "ai" and self.landmark_detector is not None and mask_vis is not None:
                        try:
                            # 根据掩码计算外接矩形，得到鱼的裁剪区域
                            ys, xs = np.where(mask_vis > 0)
                            if ys.size > 0 and xs.size > 0:
                                x1, y1 = int(xs.min()), int(ys.min())
                                x2, y2 = int(xs.max())+1, int(ys.max())+1
                                crop_bgr = color_image[y1:y2, x1:x2]
                                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                                # 预测局部坐标系下的两个关键点（0=body_center, 1=head_center）
                                pred_landmarks, pred_visibility = self.landmark_detector.predict(crop_rgb)
                                if pred_landmarks.shape[0] >= 2:
                                    body_xy_local = pred_landmarks[0]
                                    head_xy_local = pred_landmarks[1]
                                else:
                                    # 兼容只有一个点的情况
                                    body_xy_local = pred_landmarks[0]
                                    head_xy_local = pred_landmarks[0]

                                # 映射回全图坐标（像素）
                                u_body = float(x1 + body_xy_local[0])
                                v_body = float(y1 + body_xy_local[1])
                                u_head = float(x1 + head_xy_local[0])
                                v_head = float(y1 + head_xy_local[1])

                                # 深度（米）
                                z_body_m = float(depth_image[int(round(v_body)), int(round(u_body))]) / 1000.0 if 0 <= int(round(v_body)) < depth_image.shape[0] and 0 <= int(round(u_body)) < depth_image.shape[1] else 0.0
                                z_head_m = float(depth_image[int(round(v_head)), int(round(u_head))]) / 1000.0 if 0 <= int(round(v_head)) < depth_image.shape[0] and 0 <= int(round(u_head)) < depth_image.shape[1] else 0.0
                                if z_body_m <= 0:
                                    print(f"无效身体中心深度: {z_body_m}")
                                    raise ValueError("无效深度")

                                # 反投影到相机坐标系（身体中心）
                                Xb = (u_body - self.cx) / self.fx * z_body_m
                                Yb = (v_body - self.cy) / self.fy * z_body_m
                                point_cam_body = np.array([[Xb, Yb, z_body_m]], dtype=np.float32)

                                # 反投影到相机坐标系（头部中心，如无效深度则沿用身体深度）
                                if z_head_m <= 0:
                                    z_head_m = z_body_m
                                Xh = (u_head - self.cx) / self.fx * z_head_m
                                Yh = (v_head - self.cy) / self.fy * z_head_m
                                point_cam_head = np.array([[Xh, Yh, z_head_m]], dtype=np.float32)

                                # 相机→夹爪
                                point_grip_body = self.apply_hand_eye_transform(point_cam_body)[0]
                                point_grip_head = self.apply_hand_eye_transform(point_cam_head)[0]
                                body_grip_mm = point_grip_body * 1000.0
                                head_grip_mm = point_grip_head * 1000.0

                                # 方向向量（图像坐标系，单位向量）
                                dir_img = np.array([u_head - u_body, v_head - v_body], dtype=np.float32)
                                norm_img = np.linalg.norm(dir_img) + 1e-6
                                dir_img_unit = (dir_img / norm_img).tolist()

                                # 方向向量（夹爪坐标系XY，单位向量，mm）
                                dir_grip_xy = np.array([head_grip_mm[0] - body_grip_mm[0], head_grip_mm[1] - body_grip_mm[1]], dtype=np.float32)
                                norm_grip = np.linalg.norm(dir_grip_xy) + 1e-6
                                dir_grip_xy_unit = (dir_grip_xy / norm_grip).tolist()


            
                                # 当前抓取按身体中心
                                delta_tool_mm = [body_grip_mm[0], body_grip_mm[1], body_grip_mm[2]]
                                delta_base_xyz = self._tool_offset_to_base(delta_tool_mm, current_tcp[3:6])
                                z_offset = -delta_tool_mm[2] - 25

                                print(f"🎯 使用AI身体中心: uv=({u_body:.1f},{v_body:.1f}) -> grip(mm)={body_grip_mm}")
                                print(f"📍 头部中心: uv=({u_head:.1f},{v_head:.1f}) -> grip(mm)={head_grip_mm}")
                                print(f"🧭 方向(像素xy,单位向量) body→head = {dir_img_unit}")
                                print(f"🧭 方向(夹爪XY,单位向量) body→head = {dir_grip_xy_unit}")
                                # 与X轴(1,0,0)的夹角（弧度），并规范化到 [-pi/2, pi/2]
                                # 这样无论鱼体原始朝向如何，都会被映射到“朝向+X半平面”的等效姿态，便于统一放置方向
                                angle_rad = float(np.arctan2(dir_grip_xy_unit[1], dir_grip_xy_unit[0]))
                                if angle_rad > np.pi/2:
                                    angle_rad -= np.pi
                                elif angle_rad < -np.pi/2:
                                    angle_rad += np.pi
                                

                                relative_move = [delta_base_xyz[0], delta_base_xyz[1], z_offset, 0, 0, 0]

                                print(f"🧮 方向与X轴的夹角(rad): {angle_rad:.4f}")
                            else:
                                print("[AI] 掩码为空，回退质心")
                        except Exception as e:
                            print(f"[AI] 预测身体中心失败，回退质心: {e}")

                    # 若AI未生成移动，使用质心点云方案
                    if relative_move is None:
                        # 质心点（夹爪系）
                        centroid = np.mean(points_gripper, axis=0)
                        print(f"夹爪坐标系点云质心: {centroid}")
                        center_gripper_mm = centroid * 1000
                        delta_tool_mm = [center_gripper_mm[0], center_gripper_mm[1], center_gripper_mm[2]]
                        delta_base_xyz = self._tool_offset_to_base(delta_tool_mm, current_tcp[3:6])
                        z_offset = -delta_tool_mm[2] -25
                        relative_move = [delta_base_xyz[0], delta_base_xyz[1], z_offset, 0, 0, 0]
                    
                    grasp_calc_time = time.time() - grasp_calc_start
                    self.timers['grasp_calculation'].append(grasp_calc_time)
                    print(f"⏱️  grasp_calculation: {grasp_calc_time:.3f}s")
                    
                    print("Step1 : 准备抓取")
                    print("相对移动量:", relative_move)
                    
                    # 估算鱼重量（在抓取前）
                    estimated_weight = 0.0
                    if self.fish_tracker is not None:
                        estimated_weight = self.estimate_fish_weight(points_gripper)
                        print(f"🐟 估算鱼重量: {estimated_weight:.3f}kg")
                    
                    # 机器人移动计时
                    robot_movement_start = time.time()
                    
                    # 执行相对移动
                    #import pdb; pdb.set_trace()
                    self.robot.set_digital_output(0, 0, 1)

                    ret = self.robot.linear_move(relative_move, 1, True, 500)
                    # if ret != 0:
                    #     print(f"机器人移动失败: {ret}")
                    #     self.robot.linear_move(original_tcp, 0 , True, 400)
                    #     self.robot.set_digital_output(0, 0, 0)
                    #     continue

                    #  robot move up of 20 cm relatively 
                    #  ret = self.robot.linear_move([current_tcp[0], current_tcp[1], current_tcp[2] -100, current_tcp[3], current_tcp[4], current_tcp[5]], 0, True, 400)
                    # if ret != 0:
                    #     print(f"机器人移动失败: {ret}")
                    #     self.robot.linear_move(original_tcp, 0 , True, 400)
                    #     self.robot.set_digital_output(0, 0, 0)
                    #     continue
                    self.robot.linear_move(original_tcp, 0 , True, 500)

                    print(f"旋转基座{angle_rad:.4f}弧度")
                    ret = self.robot.joint_move([-np.pi  * 0.6, 0, 0, 0, 0, angle_rad -  np.pi * 0.6], 1, True, 1)
                    ret = self.robot.linear_move([0, 0, -350, 0, 0, 0], 1 , True, 500)

                    self.robot.set_digital_output(0, 0, 0)
                    time.sleep(0.4)
                    ret = self.robot.linear_move([0, 0, 350, 0, 0, 0], 1 , True, 500)
                    ret = self.robot.joint_move([np.pi  * 0.6, 0, 0, 0, 0, 0], 1, True, 2)
                    ret = self.robot.joint_move([0, 0, 0, 0, 0,  np.pi * 0.6 - angle_rad], 1, True, 2)
                

                    #time.sleep(0.01)
                    #robot move back to the original position
                    self.robot.linear_move(original_tcp, 0 , True, 500)
                    
                    # 记录鱼到跟踪器
                    if self.fish_tracker is not None and estimated_weight > 0:
                        # 预测最终放置位置
                        predicted_final_pose = None
                        if self.position_solver is not None:
                            # 估算鱼尺寸（基于点云边界框）
                            if points_gripper is not None and len(points_gripper) > 0:
                                min_coords = np.min(points_gripper, axis=0)
                                max_coords = np.max(points_gripper, axis=0)
                                fish_size_mm = (max_coords - min_coords) * 1000.0  # 转换为mm
                                
                                # 获取下一个鱼ID
                                next_fish_id = self.fish_tracker.current_fish_id + 1
                                
                                # 预测放置位置
                                placement = self.position_solver.find_optimal_position(
                                    fish_id=next_fish_id,
                                    fish_size_mm=fish_size_mm
                                )
                                
                                if placement:
                                    # 将容器坐标转换为机器人坐标系
                                    # 假设容器在机器人工作空间中的位置
                                    container_offset = [500.0, 0.0, 100.0]  # 容器在机器人坐标系中的偏移
                                    predicted_final_pose = [
                                        container_offset[0] + placement.x_mm,
                                        container_offset[1] + placement.y_mm,
                                        container_offset[2] + placement.z_mm,
                                        0.0, 0.0, 0.0  # 末端姿态
                                    ]
                                    print(f"📍 预测放置位置: ({placement.x_mm:.1f}, {placement.y_mm:.1f}, {placement.z_mm:.1f})mm")
                                else:
                                    print("⚠️  无法找到合适的放置位置")
                        
                        # 添加鱼记录
                        fish_id = self.fish_tracker.add_fish(
                            weight_kg=estimated_weight,
                            initial_pose=current_tcp,
                            grasp_angle=angle_rad
                        )
                        
                        # 更新鱼状态为已放置
                        processing_time = time.time() - robot_movement_start
                        self.fish_tracker.update_fish_status(
                            fish_id=fish_id,
                            status="placed",
                            final_pose=predicted_final_pose,
                            processing_time=processing_time
                        )
                        
                        # 显示容器状态
                        self.fish_tracker.print_status()
                        
                        # 显示位置求解器状态
                        if self.position_solver is not None:
                            self.position_solver.print_placement_status()
                   
                    robot_movement_time = time.time() - robot_movement_start
                    self.timers['robot_movement'].append(robot_movement_time)
                    print(f"⏱️  robot_movement: {robot_movement_time:.3f}s")
            
                else:
                    print("点云为空，跳过机器人控制")

                # 整个循环计时结束
                cycle_time = time.time() - cycle_start
                self.timers['total_cycle'].append(cycle_time)
                print(f"⏱️  total_cycle: {cycle_time:.3f}s")
                print("-" * 50)
 
                # self.robot.logout()
                # exit()
                

        except KeyboardInterrupt:
            print("\n用户中断处理")
            self.robot.set_digital_output(0, 0, 0)
        except Exception as e:
            print(f"处理过程中出错: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """
        清理资源
        """
        cv2.destroyAllWindows()
        if self.pipeline:
            self.pipeline.stop()
        
        # 打印时间统计摘要
        self.print_timing_summary()
        
        print(f"处理完成！总共处理了 {self.frame_count} 帧")
        print(f"结果保存在: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='实时人体分割和3D点云生成')
    parser.add_argument('--output_dir', type=str, default='realtime_output',
                      help='输出目录路径 (默认: realtime_output)')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cpu', 'cuda'],
                      help='运行设备 (默认: cuda)')
    parser.add_argument('--save_pointcloud', action='store_true',
                      help='保存3D点云')
    parser.add_argument('--max_frames', type=int, default=None,
                      help='最大处理帧数 (默认: 无限)')
    parser.add_argument('--no_preview', action='store_true',
                      help='不显示预览窗口')
    parser.add_argument('--intrinsics_file', type=str, default=None,
                      help='相机内参JSON文件路径')
    parser.add_argument('--hand_eye_file', type=str, default=None,
                      help='手眼标定4x4齐次矩阵的.npy文件路径（相机→夹爪）')
    parser.add_argument('--bbox_selection', type=str, default='highest_confidence',
                      choices=['smallest', 'largest', 'highest_confidence'],
                      help='边界框选择策略: smallest(选择面积最小的鱼) 或 largest(选择面积最大的鱼) (默认: largest)')
    parser.add_argument('--debug', action='store_true',
                      help='启用调试模式，保存所有中间文件（RGB、深度、检测、分割、点云）')
    parser.add_argument('--use_yolo', action='store_true',
                      help='使用YOLO作为检测器（替代Grounding DINO）')
    parser.add_argument('--yolo_weights', type=str, default=None,
                      help='YOLO权重文件(.pt)路径（与 --use_yolo 搭配使用）')
    parser.add_argument('--grasp_point_mode', type=str, default='centroid',
                      choices=['centroid', 'ai'],
                      help='抓取点模式: centroid(点云质心) 或 ai(使用AI身体中心)')
    parser.add_argument('--landmark_model_path', type=str, default=None,
                      help='AI身体中心模型路径 (.pth)，当 grasp_point_mode=ai 时必需')
    parser.add_argument('--enable_weight_tracking', action='store_true',
                      help='启用鱼重量跟踪功能')
    parser.add_argument('--max_container_weight', type=float, default=12.5,
                      help='容器最大重量（kg），默认12.5kg')
    
    args = parser.parse_args()
    
    try:
        # 创建处理器
        processor = RealtimeSegmentation3D(
            output_dir=args.output_dir,
            device=args.device,
            save_pointcloud=args.save_pointcloud,
            intrinsics_file=args.intrinsics_file,
            hand_eye_file=args.hand_eye_file,
            bbox_selection=args.bbox_selection,
            debug=args.debug,
            use_yolo=args.use_yolo,
            yolo_weights=args.yolo_weights,
            grasp_point_mode=args.grasp_point_mode,
            landmark_model_path=args.landmark_model_path,
            enable_weight_tracking=args.enable_weight_tracking,
            max_container_weight=args.max_container_weight
        )
        
        # 运行实时处理
        processor.run_realtime(
            max_frames=args.max_frames,
            show_preview=not args.no_preview
        )
        
    except Exception as e:
        print(f"程序出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
