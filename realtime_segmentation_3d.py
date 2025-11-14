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
import numpy as np
import cv2
import torch
from datetime import datetime
from PIL import Image
import json

# 导入现有模块的功能
from seg import init_models
from util import (
    estimate_body_angle_alpha1,
    draw_principal_axis,
    angle_between_2d_from_origin,
    apply_hand_eye_transform as util_apply_hand_eye_transform,
    tool_offset_to_base as util_tool_offset_to_base,
)
from mask_to_3d import mask_to_3d_pointcloud, save_pointcloud, load_camera_intrinsics
from filter_mask import divide_mask
from realsense_capture import (
    setup_realsense,
    save_pointcloud_to_file,
    capture_frames as rs_capture_frames,
    capture_frames_with_retry as rs_capture_frames_with_retry,
    validate_camera_connection as rs_validate_camera_connection,
    check_camera_health as rs_check_camera_health,
)
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


class RealtimeSegmentation3D:
    def __init__(self, output_dir, device="cpu", save_pointcloud=True, intrinsics_file=None, hand_eye_file=None, bbox_selection="highest_confidence", debug=False, use_yolo=False, yolo_weights=None,
                 grasp_point_mode: str = "centroid", landmark_model_path: str = None, enable_weight_tracking: bool = True, max_container_weight: float = 12.5, det_gray: bool = False,
                 camera_calib_json: str = None, robot_config: str = "config/robot.json", erode_bbox: bool = False, erode_ratio: float = 0.1, bbox_scale: float = 1.0,
                 seg_model: str = "sam", yolo_seg_weights: str = None):
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
            erode_bbox: 是否对mask进行上下方向腐蚀（用于更精确的质心计算）
            erode_ratio: 腐蚀比例，上下各腐蚀的比例（默认0.1，即10%）
            bbox_scale: 边界框缩放因子（默认1.0，即不缩放；>1.0放大，<1.0缩小）
            seg_model: 分割模型类型 ("sam" 或 "yolov8_seg")，默认 "sam"
            yolo_seg_weights: YOLOv8分割模型权重路径（当seg_model="yolov8_seg"时必需）
        """
        self.output_dir = output_dir
        self.device = device
        self.save_pointcloud = save_pointcloud
        self.bbox_selection = bbox_selection
        self.debug = debug
        self.use_yolo = use_yolo
        self.yolo_weights = yolo_weights
        self.erode_bbox = erode_bbox
        self.erode_ratio = erode_ratio
        self.bbox_scale = bbox_scale
        # detection-only grayscale support (optional)
        self.det_gray = det_gray
        # 抓取点模式：centroid 或 ai
        self.grasp_point_mode = grasp_point_mode
        self.landmark_model_path = landmark_model_path
        # 重量跟踪相关
        self.enable_weight_tracking = enable_weight_tracking
        self.max_container_weight = max_container_weight
        # configs
        self.camera_calib_json = camera_calib_json
        self.robot_config = robot_config
        # 分割模型相关
        self.seg_model = seg_model
        self.yolo_seg_weights = yolo_seg_weights
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
        #self.sam_predictor, self.grounding_dino_model, self.processor = init_models(device)
        self.seg_predictor = init_models(device, seg_model=seg_model, yolo_seg_weights=yolo_seg_weights)
        # 为了向后兼容，保留 sam_predictor 属性
        if seg_model == "sam":
            self.sam_predictor = self.seg_predictor
        else:
            self.sam_predictor = None  # 使用yolov8_seg时不需要sam_predictor

        if self.use_yolo:
            if not self.yolo_weights or not os.path.exists(self.yolo_weights):
                print(f"[警告] 已启用YOLO检测，但未找到权重: {self.yolo_weights}，将回退Grounding DINO")
                self.use_yolo = False
        
            try:
                from ultralytics import YOLO
            except Exception as e:
                print("[错误] 未找到 ultralytics，请先: pip install ultralytics")
                print(e)
                return []

            # 加载模型（每次调用加载避免与其他依赖冲突；若频繁调用可外部缓存模型实例）
            self.yolo_model = YOLO(self.yolo_weights)
            print(f"已加载YOLO权重: {self.yolo_weights}")
                
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
        # 若未加载到，尝试从 camera_calib_json 加载，否则使用硬编码的R、t（相机→夹爪）
        if self.hand_eye_transform is None:
            loaded_json = False
            try:
                if self.camera_calib_json and os.path.exists(self.camera_calib_json):
                    with open(self.camera_calib_json, 'r', encoding='utf-8') as f:
                        calib = json.load(f)
                    he = calib.get('hand_eye', {})
                    R = np.array(he.get('R', []), dtype=np.float32)
                    t = np.array(he.get('t', []), dtype=np.float32).reshape(3, 1) if he.get('t', None) is not None else None
                    if R.shape == (3, 3) and t is not None and t.shape == (3, 1):
                        self.hand_eye_transform = np.eye(4, dtype=np.float32)
                        self.hand_eye_transform[:3, :3] = R
                        self.hand_eye_transform[:3, 3:4] = t
                        loaded_json = True
                        print(f"已从 JSON 加载手眼标定矩阵: {self.camera_calib_json}")
            except Exception as e:
                print(f"读取 camera_calib_json 失败: {e}")
            if not loaded_json:
                R_default = np.array([
                    [-0.99455141, -0.08982915, -0.05289825],
                    [ 0.09064269, -0.99579624, -0.01318166],
                    [-0.05149178, -0.01790468,  0.9985129 ]
                ], dtype=np.float32)
                t_default = np.array([[0.0607777], [0.10496735], [-0.18889416]], dtype=np.float32)

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
    
    def capture_frames(self, timeout_ms=10000):
        """委托到 realsense_capture.capture_frames"""
        return rs_capture_frames(self.pipeline, self.align, timeout_ms)
    
    def capture_frames_with_retry(self, max_retries=3, timeout_ms=10000):
        return rs_capture_frames_with_retry(self.pipeline, self.align, max_retries, timeout_ms)
    
    def validate_camera_connection(self, timeout_ms=5000):
        return rs_validate_camera_connection(self.pipeline, self.align, timeout_ms)
    
    def check_camera_health(self):
        return rs_check_camera_health(self.pipeline)
    
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
        #if getattr(self, 'use_yolo', False):
        # YOLO 路径：detect_yolo 已返回所有满足条件的框 (x1,y1,x2,y2,conf)
        boxes = self.detect_yolo(color_image, self.yolo_weights, conf=0.6, iou=0.45, imgsz=640, min_area=8000)
       
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
        
        # 根据分割模型类型选择不同的处理方式
        if self.seg_model == "yolov8_seg":
            # 使用YOLOv8分割模型
            # YOLOv8可以直接对整个图像进行分割，然后根据bbox提取对应的mask
            try:
                yolo_results = self.seg_predictor(color_image, verbose=False)
                # 提取所有检测结果
                all_yolo_boxes = []
                all_yolo_masks = []
                for result in yolo_results:
                    if result.masks is not None:
                        boxes_yolo = result.boxes.xyxy.cpu().numpy()
                        masks_yolo = result.masks.data.cpu().numpy()
                        all_yolo_boxes.extend(boxes_yolo)
                        all_yolo_masks.extend(masks_yolo)
            except Exception as e:
                print(f"[分割] YOLOv8分割失败: {e}")
                return None, None
        else:
            # 使用SAM模型
            self.sam_predictor.set_image(image_rgb)

        best_idx = -1
        best_depth_m = float('inf')
        best_mask = None
        best_confidence = -1.0  # For highest_confidence selection

        segmentation_start = time.time()
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b[:4]
            confidence = b[4] if len(b) > 4 else 0.0  # Get confidence from bbox
            
            if self.seg_model == "yolov8_seg":
                # 使用YOLOv8分割
                try:
                    # 从YOLOv8结果中找到与当前bbox最匹配的检测
                    mask_np = None
                    best_iou = 0.0
                    best_mask_idx = -1
                    
                    for j, (box_yolo, mask_yolo) in enumerate(zip(all_yolo_boxes, all_yolo_masks)):
                        # 计算IoU
                        box_yolo_x1, box_yolo_y1, box_yolo_x2, box_yolo_y2 = box_yolo
                        intersection_x1 = max(x1, box_yolo_x1)
                        intersection_y1 = max(y1, box_yolo_y1)
                        intersection_x2 = min(x2, box_yolo_x2)
                        intersection_y2 = min(y2, box_yolo_y2)
                        
                        if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                            box_area = (x2 - x1) * (y2 - y1)
                            box_yolo_area = (box_yolo_x2 - box_yolo_x1) * (box_yolo_y2 - box_yolo_y1)
                            union_area = box_area + box_yolo_area - intersection_area
                            iou = intersection_area / union_area if union_area > 0 else 0.0
                            
                            if iou > best_iou and iou > 0.3:  # 阈值0.3
                                best_iou = iou
                                best_mask_idx = j
                    
                    if best_mask_idx >= 0:
                        # 将mask转换为uint8格式，并调整到图像尺寸
                        mask_yolo = all_yolo_masks[best_mask_idx]
                        h, w = color_image.shape[:2]
                        if mask_yolo.shape != (h, w):
                            # 如果mask尺寸不匹配，需要调整
                            mask_yolo = cv2.resize(mask_yolo, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_np = (mask_yolo * 255).astype(np.uint8)
                    else:
                        print(f"[分割] 候选框 {i} 未找到匹配的YOLOv8分割结果 (IoU阈值: 0.3)")
                        continue
                except Exception as e:
                    print(f"[分割] 候选框 {i} YOLOv8分割失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                # 使用SAM分割
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

            # 应用divide_mask优化mask，提取最大连通区域
            try:
                mask_np_refined = divide_mask(mask_np, verbose=False)
                # 确保返回的mask格式一致（0/255）
                if mask_np_refined.max() <= 1:
                    mask_np_refined = mask_np_refined * 255
                mask_np = mask_np_refined.astype(np.uint8)
            except Exception as e:
                # 如果优化失败，使用原始mask
                if self.debug:
                    print(f"[优化] 候选框 {i} mask优化失败，使用原始mask: {e}")

            # 应用垂直腐蚀（如果启用）
            if self.erode_bbox:
                mask_np = self.erode_mask_vertical(mask_np)

            # 使用2D质心+深度计算深度（相机坐标系，单位米）
            mask_bool = (mask_np > 0)
            if not np.any(mask_bool):
                print(f"[分割] 候选框 {i} 掩码为空，跳过")
                continue

            # 计算2D质心
            try:
                ys_mask, xs_mask = np.where(mask_bool)
                if ys_mask.size == 0 or xs_mask.size == 0:
                    print(f"[分割] 候选框 {i} 掩码为空，跳过")
                    continue
                
                centroid_x_2d = int(np.mean(xs_mask))
                centroid_y_2d = int(np.mean(ys_mask))
                
                # 确保坐标在图像范围内
                h, w = depth_image.shape
                centroid_x_2d = max(0, min(w - 1, centroid_x_2d))
                centroid_y_2d = max(0, min(h - 1, centroid_y_2d))
                
                # 获取深度值（毫米）
                depth_mm = depth_image[int(centroid_y_2d), int(centroid_x_2d)]
                
                if depth_mm > 0:
                    # 转换为米
                    depth_m = depth_mm / 1000.0
                    print(f"候选框 {i} 2D质心: ({centroid_x_2d}, {centroid_y_2d}), 深度: {depth_m:.4f} m  bbox=({x1},{y1},{x2},{y2})")
                else:
                    print(f"[深度] 候选框 {i} 质心位置深度值为0，跳过")
                    continue
                    
            except Exception as e:
                print(f"[计算] 候选框 {i} 计算2D质心深度失败: {e}，跳过")
                continue

            # 记录调试输出
            if self.debug:
                cv2.imwrite(os.path.join(self.segmentation_dir, f"{base_name}_cand{i}_mask.png"), mask_np)
                det_vis = color_image.copy()
                cv2.rectangle(det_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(det_vis, f"cand {i}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.imwrite(os.path.join(self.detection_dir, f"{base_name}_cand{i}_box.png"), det_vis)

            # Select best bbox based on bbox_selection strategy
            should_update = False
            if self.bbox_selection == "highest_confidence":
                # Select bbox with highest confidence
                if confidence > best_confidence:
                    should_update = True
            elif self.bbox_selection == "smallest":
                # Select bbox with smallest area (closest depth)
                if depth_m < best_depth_m:
                    should_update = True
            elif self.bbox_selection == "largest":
                # Select bbox with largest area (farthest depth)
                if depth_m > best_depth_m:
                    should_update = True
            else:
                # Default: smallest depth (closest)
                if depth_m < best_depth_m:
                    should_update = True
            
            if should_update:
                best_depth_m = depth_m
                best_mask = mask_np
                best_idx = i
                best_confidence = confidence

        segmentation_time = time.time() - segmentation_start
        self.timers['segmentation'].append(segmentation_time)
        print(f"⏱️  segmentation(all): {segmentation_time:.3f}s")

        if best_idx == -1 or best_mask is None:
            print("分割/点云均失败，未选出候选")
            return None, None
        
        if self.bbox_selection == "highest_confidence":
            print(f"选择最高置信度候选: idx={best_idx}, 置信度={best_confidence:.3f}, 深度={best_depth_m:.4f} m")
        elif self.bbox_selection == "smallest":
            print(f"选择最近候选: idx={best_idx}, 深度={best_depth_m:.4f} m")
        elif self.bbox_selection == "largest":
            print(f"选择最远候选: idx={best_idx}, 深度={best_depth_m:.4f} m")
        else:
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


    def detect_yolo(self, color_image, yolo_weights_path, conf=0.25, iou=0.45, imgsz=640, min_area=1000):
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
     

        # YOLO支持直接传入numpy图像；确保为RGB
        try:
            # grayscale only for detection if enabled
            det_input = color_image
            if getattr(self, 'det_gray', False):
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                det_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            results = self.yolo_model.predict(
                source=[det_input],
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
            
            # 应用边界框缩放（在YOLO检测后立即应用）
            x1, y1, x2, y2 = self.scale_bbox(x1, y1, x2, y2, H, W)
            
            # 确保坐标在图像范围内（缩放后再次检查）
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
        """使用 util.apply_hand_eye_transform 应用手眼标定变换"""
        return util_apply_hand_eye_transform(points, self.hand_eye_transform)
    
    def pixel_to_3d_camera(self, u, v, depth_mm):
        """
        将2D像素坐标和深度值转换为3D相机坐标系坐标
        
        Args:
            u: 像素x坐标
            v: 像素y坐标
            depth_mm: 深度值（毫米）
        
        Returns:
            point_3d: 3D点坐标 (X, Y, Z) 单位：米（相机坐标系）
        """
        # 转换深度单位为米
        z_m = depth_mm / 1000.0
        
        # 使用相机内参将像素坐标转换为3D坐标
        # X = (u - cx) / fx * z
        # Y = (v - cy) / fy * z
        # Z = z
        X = (u - self.cx) / self.fx * z_m
        Y = (v - self.cy) / self.fy * z_m
        Z = z_m
        
        return np.array([X, Y, Z], dtype=np.float32)
    
    def scale_bbox(self, x1, y1, x2, y2, image_height, image_width):
        """
        从中心点缩放边界框
        
        Args:
            x1, y1, x2, y2: 原始边界框坐标
            image_height: 图像高度
            image_width: 图像宽度
        
        Returns:
            scaled_x1, scaled_y1, scaled_x2, scaled_y2: 缩放后的边界框坐标
        """
        if self.bbox_scale == 1.0:
            return x1, y1, x2, y2
        
        # 计算中心点和尺寸
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # 缩放尺寸
        new_width = width * self.bbox_scale
        new_height = height * self.bbox_scale
        
        # 计算新的边界框坐标
        scaled_x1 = int(center_x - new_width / 2.0)
        scaled_y1 = int(center_y - new_height / 2.0)
        scaled_x2 = int(center_x + new_width / 2.0)
        scaled_y2 = int(center_y + new_height / 2.0)
        
        # 裁剪到图像范围内
        scaled_x1 = max(0, min(scaled_x1, image_width - 1))
        scaled_y1 = max(0, min(scaled_y1, image_height - 1))
        scaled_x2 = max(0, min(scaled_x2, image_width - 1))
        scaled_y2 = max(0, min(scaled_y2, image_height - 1))
        
        # 确保 x2 > x1 和 y2 > y1
        if scaled_x2 <= scaled_x1:
            scaled_x2 = scaled_x1 + 1
        if scaled_y2 <= scaled_y1:
            scaled_y2 = scaled_y1 + 1
        
        if self.debug and self.bbox_scale != 1.0:
            print(f"[缩放] 原始bbox: ({x1}, {y1}, {x2}, {y2}), 缩放后: ({scaled_x1}, {scaled_y1}, {scaled_x2}, {scaled_y2}), 缩放因子: {self.bbox_scale}")
        
        return scaled_x1, scaled_y1, scaled_x2, scaled_y2
    
    def erode_mask_vertical(self, mask):
        """
        对mask进行上下方向的腐蚀，去除边界区域以获得更精确的质心
        
        Args:
            mask: 二值mask（numpy数组，0/255格式）
        
        Returns:
            eroded_mask: 腐蚀后的mask
        """
        if not self.erode_bbox:
            return mask
        
        try:
            # 转换为二值格式（0/1）
            mask_bool = (mask > 0).astype(np.uint8)
            
            # 找到mask的有效区域（非零区域）
            ys, xs = np.where(mask_bool > 0)
            if ys.size == 0 or xs.size == 0:
                return mask
            
            y_min = int(ys.min())
            y_max = int(ys.max())
            height = y_max - y_min + 1
            
            # 计算上下各腐蚀的像素数
            erode_pixels = int(height * self.erode_ratio)
            
            if erode_pixels > 0 and height > erode_pixels * 2:
                # 创建腐蚀后的mask
                eroded_mask = np.zeros_like(mask_bool)
                
                # 只保留中间部分（去除上下各10%）
                y_start = y_min + erode_pixels
                y_end = y_max - erode_pixels + 1
                
                # 复制中间部分
                eroded_mask[y_start:y_end, :] = mask_bool[y_start:y_end, :]
                
                # 转换回0/255格式
                eroded_mask = eroded_mask.astype(np.uint8) * 255
                
                if self.debug:
                    print(f"[腐蚀] 原始高度: {height}, 腐蚀像素: {erode_pixels}, 保留高度: {y_end - y_start}")
                
                return eroded_mask
            else:
                # 如果mask太小，不进行腐蚀
                if self.debug:
                    print(f"[腐蚀] Mask太小（高度={height}），跳过腐蚀")
                return mask
                
        except Exception as e:
            print(f"[腐蚀] 腐蚀mask失败: {e}，返回原始mask")
            return mask

    def _rpy_to_rotation_matrix(self, rx, ry, rz):
        # 保留兼容方法但委托到 util（如后续直接调用 util，可删除此方法）
        from util import rpy_to_rotation_matrix
        return rpy_to_rotation_matrix(rx, ry, rz)

    def _tool_offset_to_base(self, delta_tool_xyz_mm, tcp_rpy):
        # 保留兼容方法但委托到 util
        dx, dy, dz = util_tool_offset_to_base(delta_tool_xyz_mm, tcp_rpy)
        return [dx, dy, dz]

    def calculate_pointcloud_bbox(self, points):
        from point_cloud_utils import calculate_pointcloud_bbox
        return calculate_pointcloud_bbox(points)

    def calculate_surface_normal(self, points, method='pca'):
        from point_cloud_utils import calculate_surface_normal
        return calculate_surface_normal(points, method)

    def _simple_plane_fitting(self, points, centroid):
        from point_cloud_utils import _simple_plane_fitting
        return _simple_plane_fitting(points, centroid)

    def _nearest_neighbors_normal(self, points, centroid, k=20):
        from point_cloud_utils import _nearest_neighbors_normal
        return _nearest_neighbors_normal(points, centroid, k)

    def normal_to_rpy(self, normal_vector, current_rpy=None):
        from pose import normal_to_rpy as pose_normal_to_rpy
        return pose_normal_to_rpy(normal_vector, current_rpy)

    def _smooth_rpy_transition(self, current_rpy, target_rpy, max_change=0.1):
        from pose import smooth_rpy_transition
        return smooth_rpy_transition(current_rpy, target_rpy, max_change)

    def estimate_fish_weight(self, points_gripper, volume_factor: float = 1.0) -> float:
        from util import estimate_fish_weight
        return estimate_fish_weight(points_gripper, volume_factor)

    def calculate_grasp_pose_with_normal(self, points_gripper, current_tcp):
        from pose import calculate_grasp_pose_with_normal as pose_calculate_grasp_pose_with_normal
        return pose_calculate_grasp_pose_with_normal(points_gripper, current_tcp)
    
    
    def show_preview(self, color_image, depth_image, mask, detection_vis=None, landmark_vis=None):
        """
        在一个窗口中显示2x2网格：RGB、检测、分割和关键点预测结果
        """
 
        # 调整图像大小
        display_size = (640, 480)
        
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
        print("按空格键暂停/继续")
        if self.fish_tracker is not None:
            print("按 'r' 键重置容器")
            print("按 's' 键显示状态")
            print("按 'e' 键导出数据")
        
        # 暂停标志
        paused = False
        
        # 验证相机连接
        print("[调试] 开始验证相机连接...")
        if not self.validate_camera_connection():
            print("❌ 相机连接验证失败，请检查相机连接后重试")
            return
        print("[调试] 相机连接验证通过")

        try:
            if self.robot_config and os.path.exists(self.robot_config):
                with open(self.robot_config, 'r', encoding='utf-8') as f:
                    robot_conf = json.load(f)
                init = robot_conf.get('initial_pose', {})
                pose_type = init.get('type', 'joint')
                mode = int(init.get('mode', 0))
                blocking = bool(init.get('blocking', True))
                speed = float(init.get('speed', 1))
                if pose_type == 'joint':
                    vals_deg = init.get('values_deg')
                    vals_rad = init.get('values_rad')
                    if vals_rad is not None:
                        joints = [float(x) for x in vals_rad]
                    elif vals_deg is not None:
                        joints = [float(x) * np.pi / 180.0 for x in vals_deg]
                    else:
                        joints = None
                    if joints is not None and len(joints) == 6:
                        ret = self.robot.joint_move(joints, mode, blocking, speed)
                        time.sleep(0.2)
                elif pose_type == 'tcp':
                    tcp = init.get('values_mmrad')  # [x,y,z,rx,ry,rz]
                    if tcp is not None and len(tcp) == 6:
                        ret = self.robot.linear_move(tcp, mode, blocking, speed)
                        time.sleep(0.2)
        except Exception as e:
            print(f"加载/移动初始位姿失败: {e}")
            import traceback
            traceback.print_exc()

        try:
            tcp_result = self.robot.get_tcp_position()
            joint_result = self.robot.get_joint_position()
            original_joint = []
            if isinstance(tcp_result, tuple) and len(tcp_result) == 2:
                tcp_ok, original_tcp = tcp_result
                original_joint =  joint_result[1]
            else:
                # 如果只返回一个值，假设它是位置信息
                original_tcp = tcp_result
                tcp_ok = True
            print(f"[调试] TCP位置获取成功: {original_tcp}")
            print(f"[调试] Joint位置获取成功: {original_joint}")
        except Exception as e:
            print(f"[错误] 获取TCP位置失败: {e}")
            import traceback
            traceback.print_exc()
            return  # 如果无法获取TCP位置，提前返回

        print("[调试] 开始加载鱼路径配置...")
        fish_count = 0 # count the number of fish in the container
        rows = -1
        cols = -1
        fish_path_json = {}
        
        # 始终从 fish_grid_params.json 加载配置
        grid_params_path = "configs/fish_grid_params.json"
        print(f"[调试] 从 fish_grid_params.json 加载配置...")
        
        try:
            with open(grid_params_path, 'r', encoding='utf-8') as f:
                grid_params = json.load(f)
            
            # 从配置文件获取路径文件路径
            fish_paths_path = grid_params.get('output', {}).get('waypoints_json_path')
            if not fish_paths_path:
                raise ValueError("fish_grid_params.json 中未找到 'output.waypoints_json_path' 字段")
            print(f"[调试] 从 fish_grid_params.json 读取路径: {fish_paths_path}")
            
            # 获取网格配置
            grid = grid_params.get('grid', {})
            rows = int(grid.get('rows', 0))
            cols = int(grid.get('cols', 0))
            print(f"[调试] 网格配置: rows={rows}, cols={cols}")
            
            # 加载路径文件
            print(f"[调试] 读取鱼路径文件: {fish_paths_path}")
            with open(fish_paths_path, 'r', encoding='utf-8') as f:
                fish_path_json = json.load(f)
            print(f"[调试] 成功加载鱼路径配置，包含 {len(fish_path_json)} 个位置")
            
            # 如果 rows 或 cols 无效，尝试从路径文件推断
            if rows <= 0 or cols <= 0:
                total_positions = len(fish_path_json)
                if rows <= 0 and cols > 0:
                    rows = max(1, total_positions // cols)
                elif cols <= 0 and rows > 0:
                    cols = max(1, total_positions // rows)
                else:
                    # 如果都不知道，假设是单列
                    rows = total_positions
                    cols = 1
                print(f"[调试] 从路径文件推断 rows:{rows}, cols:{cols}")
            
        except FileNotFoundError:
            print(f"[错误] 配置文件不存在: {grid_params_path}")
            print(f"[错误] 请确保 {grid_params_path} 文件存在")
            import traceback
            traceback.print_exc()
        except KeyError as e:
            print(f"[错误] fish_grid_params.json 中缺少必要的字段: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"[错误] 加载配置失败: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            while True:
                # 处理键盘输入（包括暂停状态下的输入）
                if show_preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # 空格键
                        paused = not paused
                        if paused:
                            print("⏸️  已暂停 - 按空格键继续")
                        else:
                            print("▶️  已继续")
                    elif key == ord('q'):
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
                
                # 如果暂停，跳过处理但继续监听键盘
                if paused:
                    continue
                
                # 整个循环计时开始
                cycle_start = time.time()
                
                # 捕获帧（使用重试机制）
                color_image, depth_image, success = self.capture_frames() #self.capture_frames_with_retry(max_retries=3, timeout_ms=10000)
                if not success:
                    print("⚠️  跳过此帧，继续处理下一帧...")
                    continue
                
                self.frame_count = self.frame_count + 1
                        # 这里可以添加重新连接逻辑
                if self.frame_count < 10 :
                    print(f"跳过前10帧，等待相机稳定...")
                    if self.frame_count == 9 and show_preview:
                        self.show_preview(color_image, depth_image, None, None, None)
                        cv2.waitKey(1)
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
                    #if getattr(self, 'use_yolo', False):
                    boxes = self.detect_yolo(color_image, self.yolo_weights, conf=0.6, iou=0.45, imgsz=640)
                  
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
                
                # 生成关键点/方向可视化
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

                # 若未生成关键点可视化，则可视化主方向（PCA）
                if landmark_vis is None and mask_vis is not None:
                    try:
                        landmark_vis = draw_principal_axis(color_image, mask_vis > 0)
                    except Exception:
                        landmark_vis = None
                
                # 应用垂直腐蚀（如果启用）- 对最终选择的mask进行腐蚀
                mask_vis_eroded = mask_vis
                if mask_vis is not None and self.erode_bbox:
                    mask_vis_eroded = self.erode_mask_vertical(mask_vis.copy())
                
                # 计算2D质心（用于后续3D转换）
                centroid_2d = None
                if mask_vis_eroded is not None:
                    try:
                        # 计算掩码的质心（使用腐蚀后的mask）
                        ys, xs = np.where(mask_vis_eroded > 0)
                        if ys.size > 0 and xs.size > 0:
                            centroid_x = int(np.mean(xs))
                            centroid_y = int(np.mean(ys))
                            centroid_2d = (centroid_x, centroid_y)
                    except Exception as e:
                        print(f"[计算] 计算2D质心失败: {e}")
                
                # 在landmark_vis上绘制质心十字标记
                if landmark_vis is not None and centroid_2d is not None:
                    try:
                        centroid_x, centroid_y = centroid_2d
                        # 绘制十字标记
                        cross_size = 25
                        cross_thickness = 4
                        cross_color = (0, 255, 255)  # 黄色 (BGR)
                        
                        # 水平线
                        cv2.line(landmark_vis, 
                               (centroid_x - cross_size, centroid_y), 
                               (centroid_x + cross_size, centroid_y), 
                               cross_color, cross_thickness)
                        # 垂直线
                        cv2.line(landmark_vis, 
                               (centroid_x, centroid_y - cross_size), 
                               (centroid_x, centroid_y + cross_size), 
                               cross_color, cross_thickness)
                    except Exception as e:
                        print(f"[可视化] 绘制质心标记失败: {e}")
                
                # 显示预览窗口
                self.show_preview(color_image, depth_image, mask_vis, detection_vis, landmark_vis)
                
                # 确保窗口显示并处理按键（非暂停状态下的额外按键处理）
                if show_preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # 空格键
                        paused = not paused
                        if paused:
                            print("⏸️  已暂停 - 按空格键继续")
                        else:
                            print("▶️  已继续")
                    elif key == ord('q'):
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

                # 根据掩码生成3D点云并保存（可选应用手眼标定）
                # 注意：现在主要用于可视化，3D质心计算改为使用2D质心+深度
                points_gripper = None  # 初始化变量
                if mask_vis is not None and base_name is not None:
                    # 点云生成计时（仅用于可视化，不用于质心计算）
                    pointcloud_start = time.time()
                    mask_bool = (mask_vis > 0)
                    points, colors = self.generate_pointcloud(color_image, depth_image, mask_bool)
                    pointcloud_time = time.time() - pointcloud_start
                    self.timers['pointcloud_generation'].append(pointcloud_time)
                    print(f"⏱️  pointcloud_generation: {pointcloud_time:.3f}s")
                    if len(points) > 0:
                        # 应用手眼变换：相机→夹爪
                        points_gripper = self.apply_hand_eye_transform(points)
                        
                        # 保存点云（仅在debug模式下）
                        # if self.debug:
                        #     # 保存相机坐标系点云
                        #     # cam_ply = os.path.join(self.pointcloud_dir, f"{base_name}_cam_pointcloud.ply")
                        #     # save_pointcloud_to_file(points, colors, cam_ply)
                        #     # 保存夹爪坐标系点云
                        #     grip_ply = os.path.join(self.pointcloud_dir, f"{base_name}_gripper_pointcloud.ply")
                        #     save_pointcloud_to_file(points_gripper, colors, grip_ply)
                
                # don't forget to transform the units, the point cloud is in meter, but robot
                # control would like to be in mm. 

                # 使用2D质心+深度计算3D质心（在夹爪坐标系中）
                if centroid_2d is not None and mask_vis is not None:
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
                    alpha_1 = None  # angle between head→body (image) and vertical (image y-axis)
                    # 先用掩码基于PCA估计一个 alpha_1 作为默认值
                    try:
                        if mask_vis is not None:
                            alpha_1 = estimate_body_angle_alpha1(mask_vis > 0)
                            print(f"[PCA] 估计 alpha_1(rad) = {alpha_1:.4f}, deg = {np.degrees(alpha_1):.2f}")
                    except Exception as e:
                        print(f"[PCA] 估计 alpha_1 失败: {e}")

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

                                # 方向向量（图像坐标系，单位向量） head→body
                                dir_img = np.array([u_body - u_head, v_body - v_head], dtype=np.float32)
                                norm_img = np.linalg.norm(dir_img) + 1e-6
                                dir_img_unit = (dir_img / norm_img).tolist()

                                # 与图像竖直轴(即y轴)的夹角：使用 atan2(vx, vy)
                                # 返回[-pi, pi] 的有符号角度
                                alpha_1 = float(np.arctan2(dir_img[0], dir_img[1]))

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
                                print(f"🧭 方向(像素xy,单位向量) head→body = {dir_img_unit}")
                                print(f"📐 与图像竖直轴的夹角 alpha_1(rad) = {alpha_1:.4f}, deg = {np.degrees(alpha_1):.2f}")
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

                    # 若AI未生成移动，使用2D质心+深度方案
                    if relative_move is None:
                        try:
                            # 获取2D质心位置的深度值
                            centroid_x, centroid_y = centroid_2d
                            
                            # 确保坐标在图像范围内
                            h, w = depth_image.shape
                            centroid_x = max(0, min(w - 1, centroid_x))
                            centroid_y = max(0, min(h - 1, centroid_y))
                            
                            # 获取深度值（毫米）
                            depth_mm = depth_image[int(centroid_y), int(centroid_x)]
                            
                            if depth_mm > 0:
                                # 将2D质心+深度转换为3D相机坐标
                                centroid_camera = self.pixel_to_3d_camera(centroid_x, centroid_y, depth_mm)
                                print(f"2D质心: ({centroid_x}, {centroid_y}), 深度: {depth_mm:.1f}mm")
                                print(f"相机坐标系3D质心: {centroid_camera}")
                                
                                # 应用手眼标定转换到夹爪坐标系
                                # 需要将单个点转换为点数组格式
                                centroid_camera_array = centroid_camera.reshape(1, 3)
                                centroid_gripper_array = self.apply_hand_eye_transform(centroid_camera_array)
                                centroid_gripper = centroid_gripper_array[0]  # 提取单个点
                                
                                print(f"夹爪坐标系3D质心: {centroid_gripper}")
                                
                                # 转换为毫米
                                center_gripper_mm = centroid_gripper * 1000.0
                            else:
                                print(f"⚠️ 警告: 2D质心位置深度值为0，无法计算3D质心")
                                # 如果深度无效，回退到点云质心（如果可用）
                                if points_gripper is not None and len(points_gripper) > 0:
                                    centroid = np.mean(points_gripper, axis=0)
                                    print(f"回退到点云质心: {centroid}")
                                    center_gripper_mm = centroid * 1000.0
                                else:
                                    print("⚠️ 错误: 无法计算3D质心，跳过此目标")
                                    continue
                        except Exception as e:
                            print(f"⚠️ 计算2D质心到3D转换失败: {e}")
                            # 如果转换失败，回退到点云质心（如果可用）
                            if points_gripper is not None and len(points_gripper) > 0:
                                centroid = np.mean(points_gripper, axis=0)
                                print(f"回退到点云质心: {centroid}")
                                center_gripper_mm = centroid * 1000.0
                            else:
                                print("⚠️ 错误: 无法计算3D质心，跳过此目标")
                                continue
                        
                        delta_tool_mm = [center_gripper_mm[0], center_gripper_mm[1], center_gripper_mm[2]]
                        delta_base_xyz = self._tool_offset_to_base(delta_tool_mm, current_tcp[3:6])
                        z_offset = -delta_tool_mm[2]
                        relative_move = [delta_base_xyz[0], delta_base_xyz[1], z_offset -5, 0, 0, 0]
                        
                        # 在质心模式下，确保alpha_1的方向一致性
                        # PCA可能返回两个相反的方向，导致角度不一致（有时旋转90度）
                        # 预览图像中的箭头是正确的，说明(vx, vy)方向是正确的
                        # 我们需要确保使用的角度与预览图像一致
                        if alpha_1 is not None and mask_vis is not None:
                            try:
                                # 重新计算PCA，获取方向向量（与预览图像使用相同的计算）
                                alpha_1_raw, (vx, vy), _ = estimate_body_angle_alpha1(mask_vis > 0, return_details=True)
                                
                                # # 使用与预览图像相同的方向向量重新计算角度
                                # # 确保角度计算与预览图像一致
                                # alpha_1_new = np.arctan2(vx, vy)
                                
                                # # 将角度规范化到[0, pi]范围内
                                # # alpha_1是相对于图像y轴的角度，范围是[-pi, pi]
                                # # 我们需要将其映射到[0, pi]范围
                                # # 方法：如果角度是负的，加上2*pi，然后取模到[0, 2*pi]，最后如果超过pi，取补角
                                # if alpha_1_new < 0:
                                #     alpha_1_new = alpha_1_new + 2 * np.pi
                                # # 如果角度在[pi, 2*pi]范围内，映射到[0, pi]（取补角）
                                # if alpha_1_new > np.pi:
                                #     alpha_1_new = 2 * np.pi - alpha_1_new
                                
                                # 确保角度在[0, pi]范围内
                                alpha_1_new = (alpha_1_raw + np.pi) % np.pi 
                                
                                alpha_1 = alpha_1_new
                                print(f"[质心模式] 原始alpha_1: {alpha_1_raw:.4f} rad ({np.degrees(alpha_1_raw):.2f} deg)")
                                print(f"[质心模式] 修正后的alpha_1: {alpha_1:.4f} rad ({np.degrees(alpha_1):.2f} deg), vx={vx:.3f}, vy={vy:.3f}")
                            except Exception as e:
                                print(f"[质心模式] 修正alpha_1失败: {e}，使用原始值")
                                import traceback
                                traceback.print_exc()
                    
                    grasp_calc_time = time.time() - grasp_calc_start
                    self.timers['grasp_calculation'].append(grasp_calc_time)
                    print(f"⏱️  grasp_calculation: {grasp_calc_time:.3f}s")
                    

                    # 执行相对移动
                    fish_count += 1
                    counter = rows * cols
                    if counter <= 0:
                        print(f"[错误] 无效的网格配置: rows={rows}, cols={cols}, counter={counter}")
                        counter = len(fish_path_json) if fish_path_json else 1
                        print(f"[警告] 使用路径文件数量作为计数器: {counter}")
                    fish_count = ((fish_count - 1) % counter) + 1  # 确保 fish_count 在 1 到 counter 之间
                    print(f"[调试] fish_count={fish_count}, counter={counter}, rows={rows}, cols={cols}")

                    if fish_count == 1:
                        continue

                    if fish_count == 6:
                        continue

                    print("Step1 : 准备抓取")
                    print("相对移动量:", relative_move)
                    
                    # 估算鱼重量（在抓取前）
                    estimated_weight = 0.0
                    if self.fish_tracker is not None:
                        estimated_weight = self.estimate_fish_weight(points_gripper)
                        print(f"🐟 估算鱼重量: {estimated_weight:.3f}kg")
                    
                    # 机器人移动计时
                    robot_movement_start = time.time()
                    # delta_rad = float(alpha_1)
                    # delta_rad = np.clip(delta_rad, 0, np.pi)
                  
                    self.robot.joint_move([0,0,0,0,0,np.pi/2-alpha_1], 1, True, 1)
                    time.sleep(0.5)
                    
                    # catch fish
                    ret = self.robot.linear_move([relative_move[0], relative_move[1], 0, 0, 0, 0], 1, True, 100)
                    self.robot.set_digital_output(0, 0, 1)

                    ret = self.robot.linear_move([0, 0, relative_move[2], 0, 0, 0], 1, True, 100)

                    if fish_count == 5 or fish_count == 10:
                        relative_move[1] = float(relative_move[1]-50)

                    time.sleep(0.3)
                  
                    #ret = self.robot.linear_move(original_tcp, 0, True, 100)

                    # go back to original point
                    #self.robot.joint_move([0,0,0,0,0,-np.pi/2+alpha_1], 1, True, 1)
                    ret = self.robot.linear_move(original_tcp, 0, True, 100)

                    rotation_flag = 1 if alpha_1_raw > 0 else -1 
                    print("-----------------------------------------------------------")
                    print("rotation_flag : {}".format(rotation_flag))
                    ret = self.robot.joint_move([0,0,0,0,0,-np.pi/2 * rotation_flag],1, False,50)

                    time.sleep(0.2)
                    
                    # get target point1
                    fish_key = str(fish_count)
                    if fish_key not in fish_path_json:
                        print(f"[错误] 路径文件中不存在键 '{fish_key}'，可用键: {list(fish_path_json.keys())}")
                        print(f"[错误] 跳过此次放置")
                        continue
                    xy_path = fish_path_json[fish_key]
                    joint_pos1 = [0, 0, 0, 0, 0, 0]
                    joint_pos1[0] = xy_path[0][0]
                    joint_pos1[1] = xy_path[0][1]
                    joint_pos1[2] = 0
                    joint_pos1[3] = 0
                    joint_pos1[4] = 0
                    joint_pos1[5] = 0
                    ret = self.robot.linear_move(joint_pos1, 1, True, 400)

                    joint_pos2 = [0, 0, 0, 0, 0, 0]
                    joint_pos2[0] = xy_path[1][0]
                    joint_pos2[1] = xy_path[1][1]
                    joint_pos2[2] = -200
                    joint_pos2[3] = 0
                    joint_pos2[4] = 0
                    joint_pos2[5] = 0
                    ret = self.robot.linear_move(joint_pos2, 1, True, 400)
                    self.robot.set_digital_output(0, 1, 1)
                    time.sleep(0.2)
                    self.robot.set_digital_output(0,0,0)
                    self.robot.set_digital_output(0,1,0)
                    ret = self.robot.linear_move([0, -joint_pos2[1], 200, 0, 0, 0], 1 , True, 400)
                    
                    self.robot.linear_move(original_tcp, 0, True, 200)
                    
                    time.sleep(0.3)
                    
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
    parser.add_argument('--det_gray', action='store_true',
                      help='仅在检测阶段使用灰度图像（SAM与抓取保持RGB）')
    parser.add_argument('--grasp_point_mode', type=str, default='centroid',
                      choices=['centroid', 'ai'],
                      help='抓取点模式: centroid(点云质心) 或 ai(使用AI身体中心)')
    parser.add_argument('--landmark_model_path', type=str, default=None,
                      help='AI身体中心模型路径 (.pth)，当 grasp_point_mode=ai 时必需')
    parser.add_argument('--enable_weight_tracking', action='store_true',
                      help='启用鱼重量跟踪功能')
    parser.add_argument('--max_container_weight', type=float, default=12.5,
                      help='容器最大重量（kg），默认12.5kg')
    parser.add_argument('--camera_calib_json', type=str, default=None,
                      help='手眼标定JSON文件路径，包含 hand_eye.R 和 hand_eye.t')
    parser.add_argument('--robot_config', type=str, default='configs/robot.json',
                      help='机器人配置文件，包含初始位姿 (默认: configs/robot.json)')
    parser.add_argument('--erode_bbox', action='store_true',
                      help='对检测到的mask进行上下方向腐蚀，去除边界区域以获得更精确的质心计算')
    parser.add_argument('--erode_ratio', type=float, default=0.1,
                      help='腐蚀比例，上下各腐蚀的比例（默认0.1，即10%%）')
    parser.add_argument('--bbox_scale', type=float, default=1.0,
                      help='边界框缩放因子（默认1.0，即不缩放；>1.0放大，<1.0缩小）')
    parser.add_argument('--seg_model', type=str, default='sam',
                      choices=['sam', 'yolov8_seg'],
                      help='分割模型类型: sam(默认) 或 yolov8_seg')
    parser.add_argument('--yolo_seg_weights', type=str, default=None,
                      help='YOLOv8分割模型权重路径(.pt)，当 --seg_model=yolov8_seg 时必需')
    
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
            max_container_weight=args.max_container_weight,
            det_gray=args.det_gray,
            camera_calib_json=args.camera_calib_json,
            robot_config=args.robot_config,
            erode_bbox=args.erode_bbox,
            erode_ratio=args.erode_ratio,
            bbox_scale=args.bbox_scale,
            seg_model=args.seg_model,
            yolo_seg_weights=args.yolo_seg_weights
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
