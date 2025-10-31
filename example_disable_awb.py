#!/usr/bin/env python3
"""
RealSense相机自动白平衡控制示例

演示如何关闭RealSense相机的自动白平衡功能，并设置手动白平衡值。

使用方法:
    # 关闭自动白平衡，使用默认4600K白平衡
    python3 example_disable_awb.py
    
    # 关闭自动白平衡，使用自定义白平衡值
    python3 example_disable_awb.py --white_balance 5000
    
    # 启用自动白平衡（默认行为）
    python3 example_disable_awb.py --enable_auto_white_balance
"""

import argparse
import sys
import time
import cv2
import numpy as np
import pyrealsense2 as rs

def setup_realsense_with_awb_control(disable_auto_white_balance=True, manual_white_balance=4600):
    """
    设置RealSense相机并控制白平衡
    
    Args:
        disable_auto_white_balance: 是否关闭自动白平衡
        manual_white_balance: 手动白平衡温度值（K）
    """
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置RGB和深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        # 启动管道
        profile = pipeline.start(config)
        print("RealSense相机启动成功")
        
        # 获取设备并配置传感器选项
        device = profile.get_device()
        rgb_sensor = device.first_color_sensor()
        
        if rgb_sensor is not None:
            if disable_auto_white_balance:
                # 关闭自动白平衡
                rgb_sensor.set_option(rs.option.enable_auto_white_balance, 0)
                print(f"✓ 已关闭自动白平衡")
                
                # 设置手动白平衡
                rgb_sensor.set_option(rs.option.white_balance, manual_white_balance)
                print(f"✓ 已设置手动白平衡: {manual_white_balance}K")
            else:
                # 启用自动白平衡
                rgb_sensor.set_option(rs.option.enable_auto_white_balance, 1)
                print("✓ 已启用自动白平衡")
        else:
            print("警告: 未找到RGB传感器")
            
        return pipeline
        
    except Exception as e:
        print(f"启动RealSense相机失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='RealSense自动白平衡控制示例')
    parser.add_argument('--enable_auto_white_balance', action='store_true',
                      help='启用自动白平衡 (默认: 关闭)')
    parser.add_argument('--white_balance', type=int, default=4600,
                      help='手动白平衡温度值(K) (默认: 4600K)')
    parser.add_argument('--duration', type=int, default=10,
                      help='预览持续时间(秒) (默认: 10秒)')
    
    args = parser.parse_args()
    
    print("RealSense自动白平衡控制示例")
    print("=" * 40)
    
    # 设置相机
    pipeline = setup_realsense_with_awb_control(
        disable_auto_white_balance=not args.enable_auto_white_balance,
        manual_white_balance=args.white_balance
    )
    
    if pipeline is None:
        print("无法启动RealSense相机")
        sys.exit(1)
    
    print(f"\n开始预览 {args.duration} 秒，按 'q' 键提前退出...")
    print("注意观察图像的颜色一致性（特别是白平衡设置的效果）")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            # 检查是否超时
            if time.time() - start_time > args.duration:
                print(f"\n预览时间结束 ({args.duration}秒)")
                break
            
            # 等待帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 创建深度可视化
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # 调整大小并拼接
            color_resized = cv2.resize(color_image, (320, 240))
            depth_resized = cv2.resize(depth_colormap, (320, 240))
            combined = np.hstack((color_resized, depth_resized))
            
            # 添加文字信息
            awb_status = "自动" if args.enable_auto_white_balance else f"手动({args.white_balance}K)"
            cv2.putText(combined, f"AWB: {awb_status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined, f"Frame: {frame_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined, "Press 'q' to quit", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('RealSense AWB Control - RGB | Depth', combined)
            
            frame_count += 1
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户按 'q' 键退出")
                break
                
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        # 清理
        cv2.destroyAllWindows()
        pipeline.stop()
        print("RealSense相机已停止")
        print(f"总共处理了 {frame_count} 帧")

if __name__ == "__main__":
    main()




