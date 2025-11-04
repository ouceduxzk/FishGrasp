#!/usr/bin/env python3
"""
RealSense相机数据采集脚本

使用pyrealsense2读取Intel RealSense相机的RGB和深度数据，并保存到指定目录。

使用方法:
    python3 realsense_capture.py --output_dir captured_data --num_frames 100 --interval 0.1

参数:
    --output_dir: 输出目录路径
    --num_frames: 要捕获的帧数 (默认: 100)
    --interval: 帧间间隔时间(秒) (默认: 0.1)
    --width: RGB图像宽度 (默认: 640)
    --height: RGB图像高度 (默认: 480)
    --depth_width: 深度图像宽度 (默认: 640)
    --depth_height: 深度图像高度 (默认: 480)
    --fps: 帧率 (默认: 30)

依赖:
    pip install pyrealsense2 numpy opencv-python
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from datetime import datetime
import open3d as o3d

def depth_to_pointcloud(depth_image, color_image, fx=615.0, fy=615.0, cx=320.0, cy=240.0):
    """
    将深度图像转换为3D点云
    
    Args:
        depth_image: 深度图像 (H, W) 单位: 毫米
        color_image: RGB图像 (H, W, 3)
        fx, fy: 焦距
        cx, cy: 主点坐标
    
    Returns:
        points: 3D点坐标 (N, 3)
        colors: RGB颜色 (N, 3)
    """
    height, width = depth_image.shape
    
    # 创建网格坐标
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # 过滤有效深度值
    valid_mask = depth_image > 0
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    depths = depth_image[valid_mask]
    
    if len(depths) == 0:
        return np.array([]), np.array([])
    
    # 计算3D坐标 (使用针孔相机模型)
    z = depths / 1000.0  # 转换为米
    x = (x_coords - cx) * z / fx
    y = (y_coords - cy) * z / fy
    
    # 组合3D点
    points = np.column_stack([x, y, z])
    
    # 获取对应的RGB颜色
    colors = color_image[valid_mask]
    colors = colors.astype(np.float32) / 255.0  # 归一化到[0,1]
    
    return points, colors

def save_pointcloud_to_file(points, colors, output_path):
    """
    保存点云为PLY文件
    
    Args:
        points: 3D点坐标 (N, 3)
        colors: RGB颜色 (N, 3)
        output_path: 输出文件路径
    """
    if len(points) == 0:
        print(f"警告: 没有有效的3D点，跳过保存: {output_path}")
        return False
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存为PLY文件
    success = o3d.io.write_point_cloud(output_path, pcd)
    if success:
        print(f"  ✓ 点云保存成功: {os.path.basename(output_path)} (点数: {len(points)})")
    else:
        print(f"  ✗ 点云保存失败: {output_path}")
    
    return success

def setup_realsense(width=640, height=480, depth_width=640, depth_height=480, fps=30, disable_auto_white_balance=True, manual_white_balance=4600):
    """
    设置RealSense相机配置（带回退策略，避免V4L2格式协商失败）
    
    Args:
        width: RGB图像宽度
        height: RGB图像高度
        depth_width: 深度图像宽度
        depth_height: 深度图像高度
        fps: 帧率
        disable_auto_white_balance: 是否关闭自动白平衡
        manual_white_balance: 手动白平衡温度值（K）
    """
    pipeline = rs.pipeline()

    # 尝试的配置列表（从常见到保守）
    attempts = [
        {"color": (width, height, rs.format.bgr8, fps), "depth": (depth_width, depth_height, rs.format.z16, fps), "label": "bgr8+z16"},
        {"color": (640, 480, rs.format.bgr8, 30), "depth": (640, 480, rs.format.z16, 30), "label": "640x480@30 bgr8+z16"},
        {"color": (1280, 720, rs.format.bgr8, 30), "depth": (640, 480, rs.format.z16, 30), "label": "1280x720 color + 640x480 depth"},
        {"color": (640, 480, rs.format.yuyv, 30), "depth": (640, 480, rs.format.z16, 30), "label": "yuyv+z16"},
        {"color": None, "depth": (640, 480, rs.format.z16, 30), "label": "depth-only"},
        {"color": "auto", "depth": "auto", "label": "librealsense-auto"},
    ]

    last_error = None
    for attempt in attempts:
        try:
            config = rs.config()
            if attempt["color"] == "auto" and attempt["depth"] == "auto":
                profile = pipeline.start()  # 让librealsense自动选择可用配置
                print("RealSense相机启动成功 (自动配置)")
                # 配置传感器选项
                _configure_sensor_options(pipeline, disable_auto_white_balance, manual_white_balance)
                return pipeline, config
            if attempt["color"] is not None:
                cw, ch, cf, cfps = attempt["color"]
                config.enable_stream(rs.stream.color, cw, ch, cf, cfps)
            if attempt["depth"] is not None:
                dw, dh, df, dfps = attempt["depth"]
                config.enable_stream(rs.stream.depth, dw, dh, df, dfps)
            profile = pipeline.start(config)
            # 打印成功信息
            print(f"RealSense相机启动成功 ({attempt['label']})")
            if attempt["color"] is not None:
                print(f"RGB流: {cw}x{ch} @ {cfps}fps")
            if attempt["depth"] is not None:
                print(f"深度流: {dw}x{dh} @ {dfps}fps")
            
            # 配置传感器选项
            _configure_sensor_options(pipeline, disable_auto_white_balance, manual_white_balance)
            return pipeline, config
        except Exception as e:
            last_error = e
            try:
                pipeline.stop()
            except Exception:
                pass
            pipeline = rs.pipeline()
            print(f"尝试配置失败 ({attempt['label']}): {e}")

    print(f"启动RealSense相机失败: {last_error}")
    return None, None

def _configure_sensor_options(pipeline, disable_auto_white_balance=True, manual_white_balance=4600):
    """
    配置RealSense传感器选项
    
    Args:
        pipeline: RealSense管道对象
        disable_auto_white_balance: 是否关闭自动白平衡
        manual_white_balance: 手动白平衡温度值（K）
    """
    try:
        # 获取设备
        device = pipeline.get_active_profile().get_device()
        
        # 获取RGB传感器
        rgb_sensor = device.first_color_sensor()
        
        if rgb_sensor is not None:
            # 关闭自动白平衡
            if disable_auto_white_balance:
                try:
                    rgb_sensor.set_option(rs.option.enable_auto_white_balance, 0)
                    print(f"✓ 已关闭自动白平衡")
                except Exception as e:
                    print(f"警告: 无法关闭自动白平衡: {e}")
            
            # 设置手动白平衡
            try:
                rgb_sensor.set_option(rs.option.white_balance, manual_white_balance)
                print(f"✓ 已设置手动白平衡: {manual_white_balance}K")
            except Exception as e:
                print(f"警告: 无法设置手动白平衡: {e}")
        else:
            print("警告: 未找到RGB传感器，无法配置白平衡选项")
            
    except Exception as e:
        print(f"警告: 配置传感器选项时出错: {e}")

def capture_frames(pipeline, align, timeout_ms=10000):
    """
    捕获一帧并返回BGR彩色图与以毫米为单位的深度图。
    返回 (color_image_bgr, depth_image_mm, success)
    """
    try:
        frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, False
        color_image = np.asanyarray(color_frame.get_data())
        height, width = depth_frame.get_height(), depth_frame.get_width()
        depth_image = np.zeros((height, width), dtype=np.uint16)
        for y in range(height):
            for x in range(width):
                dist = depth_frame.get_distance(x, y)
                if dist > 0:
                    depth_image[y, x] = int(dist * 1000)
        return color_image, depth_image, True
    except rs.error as e:
        if "Frame didn't arrive within" in str(e):
            print(f"⚠️  帧超时: {e}")
            print("   可能原因: 相机连接不稳定或USB带宽不足")
        else:
            print(f"⚠️  RealSense错误: {e}")
        return None, None, False
    except Exception as e:
        print(f"❌ 捕获帧时出错: {e}")
        return None, None, False

def capture_frames_with_retry(pipeline, align, max_retries=3, timeout_ms=10000):
    for attempt in range(max_retries):
        color_image, depth_image, success = capture_frames(pipeline, align, timeout_ms)
        if success:
            if attempt > 0:
                print(f"✅ 第{attempt + 1}次尝试成功捕获帧")
            return color_image, depth_image, True
        else:
            if attempt < max_retries - 1:
                print(f"🔄 第{attempt + 1}次尝试失败，正在重试...")
                time.sleep(0.5)
            else:
                print(f"❌ 经过{max_retries}次尝试后仍然无法捕获帧")
    return None, None, False

def validate_camera_connection(pipeline, align, timeout_ms=5000):
    try:
        print("🔍 正在验证相机连接...")
        color_image, depth_image, success = capture_frames(pipeline, align, timeout_ms)
        if success and color_image is not None and depth_image is not None:
            print("✅ 相机连接正常")
            return True
        else:
            print("❌ 相机连接异常：无法获取有效帧")
            return False
    except Exception as e:
        print(f"❌ 相机连接验证失败: {e}")
        return False

def check_camera_health(pipeline):
    try:
        frames = pipeline.wait_for_frames(timeout_ms=2000)
        return frames is not None
    except Exception:
        return False

def capture_and_save(pipeline, output_dir, num_frames=100, interval=0.1, wait_for_q=False, show_preview=True, save_pointcloud=True):
    """
    捕获并保存RGB和深度图像
    
    Args:
        pipeline: RealSense管道对象
        output_dir: 输出目录
        num_frames: 要捕获的帧数
        interval: 帧间间隔时间(秒)
        wait_for_q: 是否等待按'q'键停止
        show_preview: 是否显示实时预览窗口
    """
    # 创建输出目录
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    # 如果启用点云保存，创建点云目录
    if save_pointcloud:
        pointcloud_dir = os.path.join(output_dir, "pointclouds")
        os.makedirs(pointcloud_dir, exist_ok=True)
    
    # 获取深度传感器和相机内参
    profile = pipeline.get_active_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    
    # 获取深度比例因子
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度比例因子: {depth_scale}")
    
    # 获取相机内参
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    fx = color_intrinsics.fx
    fy = color_intrinsics.fy
    cx = color_intrinsics.ppx
    cy = color_intrinsics.ppy
    print(f"相机内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # 创建对齐对象
    align = rs.align(rs.stream.color)
    
    if wait_for_q or num_frames == 0:
        print(f"开始实时捕获，按 'q' 键停止...")
        print(f"输出目录: {output_dir}")
    else:
        print(f"开始捕获 {num_frames} 帧图像...")
        print(f"输出目录: {output_dir}")
    
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
                print(f"警告: 第 {frame_count} 帧数据无效，跳过")
                continue
            
            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            
            # 检查颜色格式并转换
            if len(color_image.shape) == 3 and color_image.shape[2] == 3:
                # RealSense输出BGR格式，转换为RGB用于显示
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                # 保存时使用BGR格式（OpenCV标准）
                color_image_save = color_image
            else:
                print(f"警告: 第 {frame_count} 帧颜色格式异常: {color_image.shape}")
                continue
            
            # 使用get_distance方法获取深度数据（更高效的方法）
            height, width = depth_frame.get_height(), depth_frame.get_width()
            depth_image = np.zeros((height, width), dtype=np.uint16)
            
            # 使用向量化操作来提高效率
            for y in range(height):
                for x in range(width):
                    dist = depth_frame.get_distance(x, y)
                    if dist > 0:
                        # 将距离转换为毫米单位（RealSense通常以米为单位）
                        depth_image[y, x] = int(dist * 1000)
            
            # 检查深度数据的有效性
            if depth_image is None or depth_image.size == 0:
                print(f"警告: 第 {frame_count} 帧深度数据无效")
                continue
            
            # 保存RGB图像
            rgb_filename = f"rgb_{frame_count:06d}.png"
            rgb_path = os.path.join(rgb_dir, rgb_filename)
            #cv2.imwrite(rgb_path, color_image_save)
            
            # 保存可视化深度图像（彩色，可见）
            depth_filename = f"depth_{frame_count:06d}.png"
            depth_path = os.path.join(depth_dir, depth_filename)
            
            # 同时保存原始深度数据（16位PNG）
            depth_raw_filename = f"depth_raw_{frame_count:06d}.png"
            depth_raw_path = os.path.join(depth_dir, depth_raw_filename)
            #cv2.imwrite(depth_raw_path, depth_image.astype(np.uint16))
            
            # 保存原始深度数据为numpy数组
            depth_numpy_filename = f"depth_{frame_count:06d}.npy"
            depth_numpy_path = os.path.join(depth_dir, depth_numpy_filename)
            #np.save(depth_numpy_path, depth_image)
            
            # 生成并保存3D点云（如果启用）
            if save_pointcloud:
                points, colors = depth_to_pointcloud(depth_image, color_image_rgb, fx, fy, cx, cy)
                if len(points) > 0:
                    pointcloud_filename = f"pointcloud_{frame_count:06d}.ply"
                    pointcloud_path = os.path.join(pointcloud_dir, pointcloud_filename)
                    save_pointcloud_to_file(points, colors, pointcloud_path)
            
            # 创建可视化的深度图像（灰度）
            valid_depth = depth_image > 0
            if valid_depth.any():
                depth_min = depth_image[valid_depth].min()
                depth_max = depth_image[valid_depth].max()
                
                # 归一化到0-255范围，保存为灰度图像
                depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
                depth_normalized[valid_depth] = ((depth_image[valid_depth] - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                
                # 保存灰度深度图像
                save_success = cv2.imwrite(depth_path, depth_normalized)
                
                if save_success:
                    print(f"  ✓ 可视化深度图像保存成功: {depth_filename}")
                    print(f"  ✓ 原始深度数据保存成功: {depth_raw_filename}")
                    print(f"  ✓ Numpy数组保存成功: {depth_numpy_filename}")
                    print(f"    深度值范围: {depth_min} - {depth_max} (有效像素: {valid_depth.sum()})")
                else:
                    print(f"  ✗ 保存深度图像失败: {depth_path}")
            else:
                print("警告: 没有有效的深度值!")
                # 保存黑色图像
                black_image = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
                cv2.imwrite(depth_path, black_image)
            
            # 打印深度值范围用于调试（仅第一帧）
            if frame_count == 0:
                print(f"原始深度图像形状: {depth_image.shape}")
                print(f"原始深度图像数据类型: {depth_image.dtype}")
                print(f"原始深度图像值范围: {depth_image.min()} - {depth_image.max()}")
                
                # 检查有效深度值
                valid_pixels = (depth_image > 0).sum()
                total_pixels = depth_image.size
                print(f"有效深度像素: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels:.1%})")
            
            # 显示实时预览
            # 将深度图像转换为可视化格式
            # 使用原始深度图像进行可视化
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
            color_display = cv2.resize(color_image_rgb, (640, 480))
            depth_display = cv2.resize(depth_colormap, (640, 480))
            
            # 水平拼接RGB和深度图像
            combined = np.hstack((color_display, depth_display))
            
            # 计算当前FPS
            current_time = time.time()
            if frame_count > 0:
                elapsed_time = current_time - start_time
                current_fps = frame_count / elapsed_time
            else:
                current_fps = 0
            
            # 添加文字说明
            cv2.putText(combined, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, f"FPS: {current_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Press 'q' to stop", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('RealSense Capture - RGB | Depth', combined)
            
            # 计算FPS
            current_time = time.time()
            fps_frame_count += 1
            
            # 每秒更新一次FPS显示
            if current_time - fps_start_time >= 1.0:
                fps = fps_frame_count / (current_time - fps_start_time)
                print(f"已保存第 {frame_count+1} 帧: RGB={rgb_filename}, Depth={depth_filename} | FPS: {fps:.1f}")
                fps_start_time = current_time
                fps_frame_count = 0
            else:
                print(f"已保存第 {frame_count+1} 帧: RGB={rgb_filename}, Depth={depth_filename}")
            
            frame_count += 1
            
            # 检查是否达到指定帧数（如果设置了的话）
            if num_frames > 0 and frame_count >= num_frames:
                print(f"\n已达到指定帧数 {num_frames}，停止捕获")
                break
            
            # 检查按键 - 使用更短的等待时间以保持高帧率
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户按 'q' 键停止捕获")
                break
            
            # 只有在指定了间隔且不是无限循环模式时才添加延迟
            if interval > 0 and num_frames > 0:
                time.sleep(interval)
                
    except KeyboardInterrupt:
        print("\n用户中断捕获")
    except Exception as e:
        print(f"捕获过程中出错: {e}")
    finally:
        # 关闭所有窗口
        cv2.destroyAllWindows()
        # 停止管道
        pipeline.stop()
        print("RealSense相机已停止")
        print(f"总共捕获了 {frame_count} 帧图像")

def main():
    parser = argparse.ArgumentParser(description='RealSense相机数据采集')
    parser.add_argument('--output_dir', type=str, default='captured_data',
                      help='输出目录路径 (默认: captured_data)')
    parser.add_argument('--num_frames', type=int, default=0,
                      help='要捕获的帧数 (默认: 0表示无限循环)')
    parser.add_argument('--interval', type=float, default=0.1,
                      help='帧间间隔时间(秒) (默认: 0.1)')
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
    parser.add_argument('--wait_for_q', action='store_true',
                      help='等待按q键停止，而不是按帧数停止')
    parser.add_argument('--no_pointcloud', action='store_true',
                      help='禁用3D点云生成和保存')
    parser.add_argument('--enable_auto_white_balance', action='store_true',
                      help='启用自动白平衡 (默认: 关闭)')
    parser.add_argument('--white_balance', type=int, default=4600,
                      help='手动白平衡温度值(K) (默认: 4600K)')
    
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
        fps=args.fps,
        disable_auto_white_balance=not args.enable_auto_white_balance,
        manual_white_balance=args.white_balance
    )
    
    if pipeline is None:
        print("无法启动RealSense相机")
        sys.exit(1)
    
    # 捕获并保存图像
    capture_and_save(pipeline, args.output_dir, args.num_frames, args.interval, args.wait_for_q, show_preview=True, save_pointcloud=not args.no_pointcloud)
    
    print(f"\n数据采集完成!")
    print(f"RGB图像保存在: {os.path.join(args.output_dir, 'rgb')}")
    print(f"深度图像保存在: {os.path.join(args.output_dir, 'depth')}")
    if not args.no_pointcloud:
        print(f"3D点云保存在: {os.path.join(args.output_dir, 'pointclouds')}")
    else:
        print("3D点云生成已禁用")

if __name__ == "__main__":
    main()
