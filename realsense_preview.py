#!/usr/bin/env python3
"""
RealSense实时预览脚本

专门用于显示实时30fps的RGB和深度图像，不保存任何文件。
专注于高帧率显示和流畅的用户体验。

使用方法:
    python3 realsense_preview.py

依赖:
    pip install pyrealsense2 numpy opencv-python
"""

import argparse
import sys
import time
import numpy as np
import cv2
import pyrealsense2 as rs

def setup_realsense(width=640, height=480, depth_width=640, depth_height=480, fps=30):
    """设置RealSense相机配置"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置RGB流
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    # 配置深度流
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
    
    # 启动管道
    try:
        profile = pipeline.start(config)
        print(f"RealSense相机启动成功")
        print(f"RGB流: {width}x{height} @ {fps}fps")
        print(f"深度流: {depth_width}x{depth_height} @ {fps}fps")
        return pipeline, config
    except Exception as e:
        print(f"启动RealSense相机失败: {e}")
        return None, None

def get_depth_array_fast(depth_frame):
    """快速获取深度数组"""
    height, width = depth_frame.get_height(), depth_frame.get_width()
    depth_array = np.zeros((height, width), dtype=np.uint16)
    
    # 使用向量化操作来提高效率
    for y in range(height):
        for x in range(width):
            dist = depth_frame.get_distance(x, y)
            if dist > 0:
                depth_array[y, x] = int(dist * 1000)  # 转换为毫米
    
    return depth_array

def main():
    parser = argparse.ArgumentParser(description='RealSense实时预览')
    parser.add_argument('--width', type=int, default=640,
                      help='RGB图像宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480,
                      help='RGB图像高度 (默认: 480)')
    parser.add_argument('--depth_width', type=int, default=640,
                      help='深度图像宽度 (默认: 640)')
    parser.add_argument('--depth_height', type=int, default=480,
                      help='深度图像高度 (默认: 480)')
    parser.add_argument('--fps', type=int, default=30,
                      help='帧率 (默认: 30)')
    parser.add_argument('--display_width', type=int, default=1280,
                      help='显示窗口宽度 (默认: 1280)')
    parser.add_argument('--display_height', type=int, default=480,
                      help='显示窗口高度 (默认: 480)')
    
    args = parser.parse_args()
    
    # 检查pyrealsense2是否可用
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("错误: 未安装pyrealsense2")
        print("请安装: pip install pyrealsense2")
        sys.exit(1)
    
    # 设置RealSense相机
    pipeline, config = setup_realsense(
        width=args.width,
        height=args.height,
        depth_width=args.depth_width,
        depth_height=args.depth_height,
        fps=args.fps
    )
    
    if pipeline is None:
        print("无法启动RealSense相机")
        sys.exit(1)
    
    # 获取相机内参
    profile = pipeline.get_active_profile()
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    fx = color_intrinsics.fx
    fy = color_intrinsics.fy
    cx = color_intrinsics.ppx
    cy = color_intrinsics.ppy
    print(f"相机内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # 创建对齐对象
    align = rs.align(rs.stream.color)
    
    print(f"开始实时预览，按 'q' 键退出...")
    print(f"显示分辨率: {args.display_width}x{args.display_height}")
    
    frame_count = 0
    start_time = time.time()
    fps_start_time = start_time
    fps_frame_count = 0
    
    try:
        while True:
            # 等待新的帧
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到RGB帧
            aligned_frames = align.process(frames)
            
            # 获取对齐后的帧
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            
            # 检查颜色格式并转换
            if len(color_image.shape) == 3 and color_image.shape[2] == 3:
                # RealSense输出BGR格式，转换为RGB用于显示
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            else:
                continue
            
            # 获取深度数据
            depth_image = get_depth_array_fast(depth_frame)
            
            # 创建深度可视化
            valid_depth = depth_image > 0
            if valid_depth.any():
                depth_min = depth_image[valid_depth].min()
                depth_max = depth_image[valid_depth].max()
                # 归一化到0-255范围
                depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
                depth_normalized[valid_depth] = ((depth_image[valid_depth] - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            else:
                depth_colormap = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
            
            # 调整图像大小以便显示
            display_width = args.display_width // 2  # 每个图像占一半宽度
            display_height = args.display_height
            
            color_display = cv2.resize(color_image_rgb, (display_width, display_height))
            depth_display = cv2.resize(depth_colormap, (display_width, display_height))
            
            # 水平拼接RGB和深度图像
            combined = np.hstack((color_display, depth_display))
            
            # 计算FPS
            current_time = time.time()
            if frame_count > 0:
                elapsed_time = current_time - start_time
                current_fps = frame_count / elapsed_time
            else:
                current_fps = 0
            
            # 添加文字说明
            cv2.putText(combined, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, f"FPS: {current_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Press 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示深度信息
            if valid_depth.any():
                valid_pixels = valid_depth.sum()
                total_pixels = depth_image.size
                valid_ratio = valid_pixels / total_pixels
                cv2.putText(combined, f"Depth: {depth_min}-{depth_max}mm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(combined, f"Valid: {valid_ratio:.1%}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('RealSense Preview - RGB | Depth', combined)
            
            # 计算FPS统计
            current_time = time.time()
            fps_frame_count += 1
            
            # 每秒更新一次FPS显示
            if current_time - fps_start_time >= 1.0:
                fps = fps_frame_count / (current_time - fps_start_time)
                print(f"当前FPS: {fps:.1f} | 帧数: {frame_count+1} | 有效深度像素: {valid_depth.sum() if valid_depth.any() else 0}")
                fps_start_time = current_time
                fps_frame_count = 0
            
            frame_count += 1
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户按 'q' 键退出")
                break
                
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"预览过程中出错: {e}")
    finally:
        # 关闭所有窗口
        cv2.destroyAllWindows()
        # 停止管道
        pipeline.stop()
        print("RealSense相机已停止")
        print(f"总共显示了 {frame_count} 帧")

if __name__ == "__main__":
    main()
