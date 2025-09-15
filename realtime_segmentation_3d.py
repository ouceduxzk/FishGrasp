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

class RealtimeSegmentation3D:
    def __init__(self, output_dir, device="cpu", save_pointcloud=True, intrinsics_file=None, hand_eye_file=None, bbox_selection="highest_confidence", debug=False, use_yolo=False, yolo_weights=None):
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
        """
        self.output_dir = output_dir
        self.device = device
        self.save_pointcloud = save_pointcloud
        self.bbox_selection = bbox_selection
        self.debug = debug
        self.use_yolo = use_yolo
        self.yolo_weights = yolo_weights
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


        import jkrc 
        self.robot = jkrc.RC("192.168.80.116")
        self.robot.login()   
    
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
    
    def detect_and_segment_and_dump(self, color_image):
        """
        本地完成检测->落盘->分割->落盘，返回用于显示的单通道uint8掩码（0/255）。
        只选择一条鱼进行分割，无检测时返回None。
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
        
        # 构建抓取姿态
        grasp_pose = np.array([
            centroid[0] * 1000,  # 转换为毫米
            centroid[1] * 1000,
            centroid[2] * 1000,
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
    
    def show_preview(self, color_image, depth_image, mask):
        """
        在一个窗口中显示RGB、深度图和分割结果
        """
        # 创建深度可视化
        valid_depth = depth_image > 0
        if valid_depth.any():
            depth_min = depth_image[valid_depth].min()
            depth_max = depth_image[valid_depth].max()
            depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
            if depth_max > depth_min:
                depth_normalized[valid_depth] = ((depth_image[valid_depth] - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        else:
            depth_colormap = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
        
        # 创建分割可视化
        if mask is not None:
            # 将掩码转换为彩色图像
            mask_colored = np.zeros_like(color_image)
            mask_colored[mask > 0] = [0, 255, 0]  # 绿色掩码
            # 叠加到原图上
            segmentation_vis = cv2.addWeighted(color_image, 0.7, mask_colored, 0.3, 0)
        else:
            segmentation_vis = color_image.copy()
        
        # 调整图像大小
        display_size = (400, 300)
        color_display = cv2.resize(color_image, display_size)
        depth_display = cv2.resize(depth_colormap, display_size)
        seg_display = cv2.resize(segmentation_vis, display_size)
        
        # 水平拼接三个图像
        combined = np.hstack((color_display, depth_display, seg_display))
        
        # 添加标签
        cv2.putText(combined, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, "Depth", (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, "Segmentation", (810, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, f"Frame: {self.frame_count}", (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('RGB | Depth | Segmentation', combined)

    
    def run_realtime(self, max_frames=None, show_preview=True):
        """
        运行实时处理
        
        Args:
            max_frames: 最大帧数 (None表示无限)
            show_preview: 是否显示预览窗口
        """
        print("开始实时处理...")
        print("按 'q' 键停止")
        

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
                # 跳过前3帧，让相机稳定
                # if self.frame_count < 10:
                #     print(f"跳过第 {self.frame_count + 1} 帧，等待相机稳定...")
                #     #self.frame_count += 1
                #     continue

                # 检测 + 分割 + 落盘
                mask_vis, base_name = self.detect_and_segment_and_dump(color_image)
                
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

                # 显示预览窗口
                self.show_preview(color_image, depth_image, mask_vis)
                
                # 确保窗口显示并处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户按 'q' 键停止")
                    break

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
                    
                    # 计算考虑法向量的抓取姿态
                    grasp_pose, normal_info = self.calculate_grasp_pose_with_normal(points_gripper, current_tcp)
                    
                    if normal_info is not None:
                        print("🎯 法向量对齐抓取:")
                        print(f"  质心: {normal_info['centroid']}")
                        print(f"  法向量: {normal_info['normal']}")
                        print(f"  RPY变化: {np.degrees(normal_info['rpy_change'])} 度")
                        
                        # 计算相对移动（包含姿态调整）
                        # 在夹爪坐标系中，位置变化就是目标位置（因为当前TCP在原点）
                        position_change = grasp_pose[:3]  # 目标物体在夹爪坐标系中的位置
                        
                        # 姿态变化：从当前RPY到目标RPY
                        orientation_change = grasp_pose[3:6] - current_tcp[3:6]
                        
                        # 组合相对移动
                        relative_move = np.concatenate([position_change, orientation_change])
                        
                        print(f"位置变化: {position_change} mm")
                        print(f"姿态变化: {np.degrees(orientation_change)} 度")
                        
                    else:
                        # 回退到简单质心抓取
                        print("⚠️  法向量计算失败，使用简单质心抓取")
                        centroid = np.mean(points_gripper, axis=0)
                        print(f"夹爪坐标系点云质心: {centroid}")
                        
                        # 硬编码高度为0.025m
                        hardcoded_height = 0.025  # 2.5cm
                        print(f"使用硬编码高度: {hardcoded_height:.3f}m")
                        
                        # 夹爪坐标系中的目标中心点（转换为毫米）
                        center_gripper_mm = centroid * 1000
                        
                        # 计算相对移动：从当前TCP位置移动到夹爪坐标系中的目标位置
                        delta_tool_mm = [center_gripper_mm[0], center_gripper_mm[1], hardcoded_height * 1000]
                        delta_base_xyz = self._tool_offset_to_base(delta_tool_mm, current_tcp[3:6])
                        z_offset = -(current_tcp[2] - hardcoded_height * 1000) + 200 - 20
                        relative_move = [delta_base_xyz[0], delta_base_xyz[1], z_offset, 0, 0, 0]
                    
                    grasp_calc_time = time.time() - grasp_calc_start
                    self.timers['grasp_calculation'].append(grasp_calc_time)
                    print(f"⏱️  grasp_calculation: {grasp_calc_time:.3f}s")
                    
                    print("Step1 : 准备抓取")
                    print("相对移动量:", relative_move)
                    
                    # 机器人移动计时
                    robot_movement_start = time.time()
                    
                    # 执行相对移动
                    #import pdb; pdb.set_trace()
                    #self.robot.set_digital_output(0, 0, 1)

                    ret = self.robot.linear_move(relative_move, 1, True, 50)
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

                    # 旋转基座90度 (Yaw轴旋转)
                    # 90度 = π/2 弧度 ≈ 1.57 弧度
                    # rotation_angle = math.pi / 2  # 90度
                    # ret = self.robot.joint_move([-np.pi  * 0.6, 0, 0, 0, 0, 0], 1, True, 1)

                    # self.robot.set_digital_output(0, 0, 0)
                    # time.sleep(0.4)
                    # ret = self.robot.joint_move([np.pi  * 0.6, 0, 0, 0, 0, 0], 1, True, 2)
                
                    # #time.sleep(0.01)
                    # #robot move back to the original position
                    # self.robot.linear_move(original_tcp, 0 , True, 500)
                   
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
            yolo_weights=args.yolo_weights
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
