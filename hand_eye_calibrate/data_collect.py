"""采集相机的照片和机械臂的位姿并保存成文件。
这里以intel realsense 相机为例， 其他相机数据读取可能需要对应修改。 """

import cv2
import numpy as np
import pyrealsense2 as rs
import jkrc 
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
    while True:
        # first the robot move to the correponding position
        #tcp_pos=[0,0, np.random.randint(-10, 10), 0, (np.random.random() - 0.5) / 10, (np.random.random() - 0.5) * 0.4]  
        #ret=robot.linear_move(tcp_pos,INCR,True, 20)  
        #time.sleep(2)
        #print(f"机器人移动到{tcp_pos}位置")

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                           cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("detection", color_image)  # 窗口显示，显示名为 Capture_Video

        k = cv2.waitKey(1) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
        if k == ord('s'):  # 键盘按一下s, 保存当前照片和机械臂位姿
            print(f"采集第{count}组数据...")
            ret, pose = robot.get_tcp_position()  # 获取当前机械臂状态 需要根据实际使用的机械臂获得
            post = list(pose)
            print(f"机械臂pose:{pose}")

            with open(f'{image_save_path}poses.txt', 'a+') as f:
                # 将列表中的元素用空格连接成一行
                pose_ = [str(i) for i in pose]
                new_line = f'{",".join(pose_)}\n'
                # 将新行附加到文件的末尾
                f.write(new_line)

            cv2.imwrite(image_save_path + str(count) + '.jpg', color_image)
            count += 1


if __name__ == "__main__":
    data_collect()
