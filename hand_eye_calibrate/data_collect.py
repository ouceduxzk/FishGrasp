"""采集相机的照片和机械臂的位姿并保存成文件。
这里以intel realsense 相机为例， 其他相机数据读取可能需要对应修改。 """

import cv2
import numpy as np
import pyrealsense2 as rs
import jkrc
import os 
robot = jkrc.RC("192.168.80.116") # Return a robot object
robot.login()
robot.enable_robot()
INCR= 1  

count = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

image_save_path = "./collect_data/"


def data_collect():
    global count
    # 确保保存目录存在
    os.makedirs(image_save_path, exist_ok=True)
    
    print("=" * 60)
    print("数据采集程序已启动")
    print("=" * 60)
    print(f"保存路径: {os.path.abspath(image_save_path)}")
    print("操作说明:")
    print("  - 按 's' 键保存当前图片和机器人位姿")
    print("  - 按 'q' 键退出程序")
    print("  - 请确保窗口 'detection' 有焦点才能捕获按键")
    print("=" * 60)
    
    cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                       cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        
        # 在图像上显示提示信息
        #cv2.putText(color_image, f"Press 's' to save (Count: {count})", 
                   #(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #cv2.putText(color_image, "Press 'q' to quit", 
                   #(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("detection", color_image)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:  # 'q' 或 ESC 键退出
            print("程序退出")
            break
        elif k == ord('s'):  # 键盘按一下s, 保存当前照片和机械臂位姿
            try:
                print(f"\n采集第{count}组数据...")
                ret, pose = robot.get_tcp_position()  # 获取当前机械臂状态
                #if not ret:
                #    print(f"错误: 无法获取机器人位姿，返回值: {ret}")
                 #   continue
                    
                print(f"机械臂pose: {pose}")
                
                # 保存位姿到文件
                poses_file = os.path.join(image_save_path, 'poses.txt')
                with open(poses_file, 'a+') as f:
                    # 将列表中的元素用逗号连接成一行
                    pose_ = [str(i) for i in pose]
                    new_line = f'{",".join(pose_)}\n'
                    f.write(new_line)
                    f.flush()  # 立即刷新到磁盘
                
                # 保存图片
                image_file = os.path.join(image_save_path, f'{count}.jpg')
                success = cv2.imwrite(image_file, color_image)
                
                if success:
                    print(f"✓ 图片已保存: {image_file}")
                    print(f"✓ 位姿已保存: {poses_file}")
                    print(f"✓ 已采集 {count + 1} 组数据\n")
                else:
                    print(f"✗ 错误: 图片保存失败: {image_file}")
                    
                count += 1
            except Exception as e:
                print(f"✗ 保存数据时发生错误: {e}")
                import traceback
                traceback.print_exc()
    
    # 清理资源
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"总共采集了 {count} 组数据")


if __name__ == "__main__":
    data_collect()
