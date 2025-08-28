import cv2
import os
import pyrealsense2 as rs
import time
import numpy as np
import sys
from datetime import datetime


def setup_realsense(width=640, height=480, fps=30):
    """
    设置RealSense相机配置
    
    Args:
        width: RGB图像宽度
        height: RGB图像高度
        fps: 帧率
    
    Returns:
        pipeline: RealSense管道对象
    """
    # 创建管道
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置RGB流
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    # 启动管道
    try:
        profile = pipeline.start(config)
        print(f"RealSense相机启动成功")
        print(f"RGB流: {width}x{height} @ {fps}fps")
        return pipeline
    except Exception as e:
        print(f"启动RealSense相机失败: {e}")
        return None

def take_single_picture(pipeline, output_path, timeout=10):
    """
    拍摄单张照片
    
    Args:
        pipeline: RealSense管道对象
        output_path: 输出文件路径
        timeout: 超时时间（秒）
    
    Returns:
        bool: 是否成功
    """
    try:
        print("等待相机稳定...")
        time.sleep(2)  # 给相机一些时间稳定
        
        print("正在拍照...")
        # 等待帧数据，设置超时
        frames = pipeline.wait_for_frames(timeout_ms=timeout*1000)
        
        # 获取RGB帧
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print("错误: 无法获取RGB帧")
            return False
        
        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        
        # 检查图像格式
        if len(color_image.shape) != 3 or color_image.shape[2] != 3:
            print(f"错误: 图像格式异常: {color_image.shape}")
            return False
        
        # 保存图像
        success = cv2.imwrite(output_path, color_image)
        
        if success:
            print(f"照片已保存: {output_path}")
            print(f"图像尺寸: {color_image.shape[1]}x{color_image.shape[0]}")
            return True
        else:
            print(f"错误: 保存图像失败: {output_path}")
            return False
            
    except rs.error as e:
        print(f"RealSense错误: {e}")
        return False
    except Exception as e:
        print(f"拍照过程中出错: {e}")
        return False

def main():
    # 创建pic目录
    output_dir = 'pic'
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now()
    
    # 设置输出文件路径
    output_path = os.path.join(output_dir, f'captured_image_{now.strftime("%Y%m%d_%H%M%S")}.png')
    
    print("正在初始化RealSense相机...")
    
    # 设置RealSense相机
    pipeline = setup_realsense(width=640, height=480, fps=30)
    
    if pipeline is None:
        print("无法启动RealSense相机")
        print("请检查:")
        print("1. RealSense相机是否正确连接")
        print("2. 是否被其他程序占用")
        print("3. 运行 'realsense-viewer' 验证相机是否正常工作")
        return
    
    try:
        # 拍摄照片
        success = take_single_picture(pipeline, output_path, timeout=10)
        
        if success:
            # 显示拍摄的照片
            image = cv2.imread(output_path)
            if image is not None:
                cv2.imshow('拍摄的照片', image)
                print("按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("无法读取保存的图像进行显示")
        else:
            print("拍照失败")
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"操作过程中出错: {e}")
    finally:
        # 停止管道
        pipeline.stop()
        print("RealSense相机已停止")

if __name__ == "__main__":
    main()