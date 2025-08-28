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
    def __init__(self, output_dir, device="cpu", save_pointcloud=True, intrinsics_file=None, hand_eye_file=None):
        """
        初始化实时分割和3D点云生成器
        
        Args:
            output_dir: 输出目录
            device: 运行设备 (cpu/cuda)
            save_pointcloud: 是否保存3D点云
            intrinsics_file: 相机内参文件路径
        """
        self.output_dir = output_dir
        self.device = device
        self.save_pointcloud = save_pointcloud
        self.hand_eye_transform = None  # 4x4 齐次矩阵，相机坐标→夹爪坐标
        
        # 创建输出目录
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
        
        # 初始化模型
        print("正在初始化AI模型...")
        self.sam_predictor, self.grounding_dino_model, self.processor = init_models(device)
        
        # 初始化RealSense相机
        print("正在初始化RealSense相机...")
        self.pipeline, self.config = setup_realsense()
        if self.pipeline is None:
            raise RuntimeError("无法启动RealSense相机")
        
        # 获取相机内参
        self.fx, self.fy, self.cx, self.cy = load_camera_intrinsics(intrinsics_file)
        print(f"使用相机内参: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        
        # 创建对齐对象
        import pyrealsense2 as rs
        self.align = rs.align(rs.stream.color)
        
        # 帧计数器
        self.frame_count = 0
        self.start_time = time.time()

        # 加载手眼标定矩阵（可选）
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
                [-0.99462885,  0.07149648,  0.07484454],
                [-0.06962775, -0.99719970,  0.02728984],
                [ 0.07658608,  0.02193200,  0.99682173]
            ], dtype=np.float32)
            t_default = np.array([[0.0247092], [0.09912939], [-0.25357213]], dtype=np.float32)
            self.hand_eye_transform = np.eye(4, dtype=np.float32)
            self.hand_eye_transform[:3, :3] = R_default
            self.hand_eye_transform[:3, 3:4] = t_default
            print("使用硬编码手眼标定矩阵 (相机→夹爪):")
            print(self.hand_eye_transform)
        
        print("初始化完成！")


        import jkrc 
        robot = jkrc.RC("192.168.80.116")
        robot.login()   
    
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
            
            return color_image, depth_image, True
            
        except Exception as e:
            print(f"捕获帧时出错: {e}")
            return None, None, False
    
    def detect_and_segment_and_dump(self, color_image):
        """
        本地完成检测->落盘->分割->落盘，返回用于显示的单通道uint8掩码（0/255）。
        无检测时返回None。
        """
        # 检测
        boxes = self._detect_boxes(color_image)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = f"frame_{self.frame_count:06d}_{timestamp}"
        det_vis = color_image.copy()
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(det_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            crop = color_image[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(os.path.join(self.detection_dir, f"{base_name}_det_{idx}.png"), crop)
        if len(boxes) > 0:
            cv2.imwrite(os.path.join(self.detection_dir, f"{base_name}_dino_detection.png"), det_vis)

        if not boxes:
            print("未检测到目标，跳过分割。")
            return None, None

        # 分割（SAM）
        try:
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image_rgb)
            boxes_tensor = torch.tensor([[x1, y1, x2, y2] for (x1, y1, x2, y2) in boxes], device=self.device)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_tensor, image_rgb.shape[:2])

            masks, scores, logits = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )

            # 为每个目标分别保存掩码（全图与裁剪）以及裁剪可视化
            per_object_masks = []
            for idx in range(masks.shape[0]):
                m_bool = masks[idx][0].cpu().numpy().astype(np.uint8)
                m_u8 = m_bool * 255
                x1, y1, x2, y2 = boxes[idx]
                # 全图掩码
                mask_full_path = os.path.join(self.segmentation_dir, f"{base_name}_obj{idx}_mask.png")
                cv2.imwrite(mask_full_path, m_u8)
                # 裁剪掩码
                mask_crop = m_u8[y1:y2, x1:x2]
                if mask_crop.size > 0:
                    mask_crop_path = os.path.join(self.segmentation_dir, f"{base_name}_obj{idx}_mask_crop.png")
                    cv2.imwrite(mask_crop_path, mask_crop)
                # 裁剪可视化
                crop = color_image[y1:y2, x1:x2]
                if crop.size > 0 and mask_crop.size > 0:
                    overlay = np.zeros_like(crop)
                    overlay[mask_crop > 0] = [0, 255, 0]
                    vis_crop = cv2.addWeighted(crop, 1.0, overlay, 0.4, 0)
                    vis_crop_path = os.path.join(self.segmentation_dir, f"{base_name}_obj{idx}_mask_crop_vis.png")
                    cv2.imwrite(vis_crop_path, vis_crop)
                per_object_masks.append(m_u8)

            # 合并掩码并保存整体可视化
            combined = np.zeros_like(per_object_masks[0], dtype=np.uint8)
            for m_u8 in per_object_masks:
                combined = np.maximum(combined, m_u8)
            mask_np = combined

            mask_path = os.path.join(self.segmentation_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, mask_np)

            colored = np.zeros_like(color_image)
            colored[mask_np > 0] = [0, 255, 0]
            vis = cv2.addWeighted(color_image, 1.0, colored, 0.4, 0)
            vis_path = os.path.join(self.segmentation_dir, f"{base_name}_mask_vis.png")
            cv2.imwrite(vis_path, vis)

            return mask_np, base_name
        except Exception as e:
            print(f"分割时出错: {e}")
            return None, None

    def _detect_boxes(self, color_image):
        """
        使用与 seg.py 相同的方式进行检测，返回bbox列表
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
            text_threshold=0.25,
            # 与 seg.py 相同的尺寸传入方式
            target_sizes=[image_pil.size[::-1]]
        )
        result = results[0]
        boxes = []
        print("\n检测结果详情:")
        print(f"检测到的目标数量: {len(result['boxes'])}")
        if len(result["boxes"]) == 0:
            return boxes
        for box in result["boxes"]:
            x1, y1, x2, y2 = [int(c) for c in box.tolist()]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            boxes.append((x1, y1, x2, y2))
        return boxes
    
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
            
            # 使用mask_to_3d_pointcloud函数
            points, colors = mask_to_3d_pointcloud(
                color_image_rgb, 
                depth_image_meters, 
                mask, 
                self.fx, self.fy, self.cx, self.cy
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
    
    def _show_preview(self, color_image, depth_image, mask):
        # 创建深度可视化
        valid_depth = depth_image > 0
        if valid_depth.any():
            depth_min = depth_image[valid_depth].min()
            depth_max = depth_image[valid_depth].max()
            depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
            depth_normalized[valid_depth] = ((depth_image[valid_depth] - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        else:
            depth_colormap = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
        
        # 创建掩码可视化
        if mask is not None:
            mask_vis = np.zeros_like(color_image)
            mask_vis[mask] = [0, 255, 0]
            mask_vis = cv2.addWeighted(color_image, 0.7, mask_vis, 0.3, 0)
        else:
            mask_vis = color_image.copy()
        
        # 绘制检测框到 det_vis
        det_vis = color_image.copy()
        for (x1, y1, x2, y2) in self._detect_boxes(color_image):
            cv2.rectangle(det_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 调整图像大小
        display_size = (320, 240)
        color_display = cv2.resize(color_image, display_size)
        depth_display = cv2.resize(depth_colormap, display_size)
        mask_display = cv2.resize(mask_vis, display_size)
        det_display = cv2.resize(det_vis, display_size)
        
        # 四宫格
        top_row = np.hstack((color_display, depth_display))
        bottom_row = np.hstack((mask_display, det_display))
        combined = np.vstack((top_row, bottom_row))
        
        # 添加文字
        cv2.putText(combined, f"Frame: {self.frame_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined, "RGB | Depth | Seg | Det", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Realtime Segmentation 3D', combined)
    
    def run_realtime(self, max_frames=None, show_preview=True):
        """
        运行实时处理
        
        Args:
            max_frames: 最大帧数 (None表示无限)
            show_preview: 是否显示预览窗口
        """
        print("开始实时处理...")
        print("按 'q' 键停止")
        
        try:
            while True:
                # 捕获帧
                color_image, depth_image, success = self.capture_frames()
                if not success:
                    continue

                # 检测 + 分割 + 落盘
                mask_vis, base_name = self.detect_and_segment_and_dump(color_image)

                # # 可视化颜色与深度
                # valid_depth = depth_image > 0
                # if valid_depth.any():
                #     depth_min = depth_image[valid_depth].min()
                #     depth_max = depth_image[valid_depth].max()
                #     depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
                #     if depth_max > depth_min:
                #         depth_normalized[valid_depth] = ((depth_image[valid_depth] - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                #     depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                # else:
                #     depth_colormap = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)

                # display_size = (640, 480)
                # color_display = cv2.resize(color_image, display_size)
                # depth_display = cv2.resize(depth_colormap, display_size)
                # combined = np.hstack((color_display, depth_display))
                # cv2.putText(combined, f"Frame: {self.frame_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                #v2.imshow('Realtime Color | Depth', combined)

                # 根据掩码生成3D点云并保存（可选应用手眼标定）
                if mask_vis is not None and base_name is not None and self.save_pointcloud:
                    mask_bool = (mask_vis > 0)
                    points, colors = self.generate_pointcloud(color_image, depth_image, mask_bool)
                    if len(points) > 0:
                        # 应用手眼变换：相机→夹爪
                        points_out = self.apply_hand_eye_transform(points)
                        pointcloud_path = os.path.join(self.pointcloud_dir, f"{base_name}_pointcloud.ply")
                        save_pointcloud_to_file(points_out, colors, pointcloud_path)
                

                # apply the transformation matrix to the pointcloud\


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
        print(f"处理完成！总共处理了 {self.frame_count} 帧")
        print(f"结果保存在: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='实时人体分割和3D点云生成')
    parser.add_argument('--output_dir', type=str, default='realtime_output',
                      help='输出目录路径 (默认: realtime_output)')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'],
                      help='运行设备 (默认: cpu)')
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
    
    args = parser.parse_args()
    
    try:
        # 创建处理器
        processor = RealtimeSegmentation3D(
            output_dir=args.output_dir,
            device=args.device,
            save_pointcloud=args.save_pointcloud,
            intrinsics_file=args.intrinsics_file,
            hand_eye_file=args.hand_eye_file
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
