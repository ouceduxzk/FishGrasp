import time
import jkrc 
import numpy as np 
robot = jkrc.RC("192.168.80.116")
robot.login()   
robot.power_on()
robot.enable_robot()

# ret = robot.joint_move([np.pi * 0.9 , 0, 0, 0, 0, 0], 1, True, 1)
ret = robot.joint_move([-193.484*np.pi/180,
                       98.108*np.pi/180, 
                       -64.836*np.pi/180, 
                       56.796*np.pi/180,
                       -270.49*np.pi/180, 
                       168.094*np.pi/180], 0, False, 0.5)

time.sleep(0.5)

# 兼容不同 JAKA SDK 版本的停止调用，依次尝试可用的方法
_stop_methods = [
    'stop',            # 常见停止
    'emergency_stop',  # 紧急停止
    'program_stop',    # 程序停止（若在控制器侧运行程序）
    'move_stop',       # 停止当前运动（部分版本）
    'motion_stop',     # 停止当前运动（部分版本）
    'motion_pause',    # 暂停（退而求其次）
]
_stopped = False
for _m in _stop_methods:
    _fn = getattr(robot, _m, None)
    if callable(_fn):
        try:
            _fn()
            _stopped = True
            break
        except Exception as e:
            print(f"调用 {_m} 失败: {e}")
if not _stopped:
    print("未找到可用的停止方法，已跳过主动停止。可用方法名示例:")
    print([name for name in dir(robot) if not name.startswith('_')])

#ret = robot.linear_move([90, -416, 0, 0, 0, 0], 1, True, 50)

# ret = robot.linear_move([0, -54, -100, 0, 0, 0], 1, True, 50)

#robot.set_digital_output(0, 0, 1)

#robot.motion_abort()

    # time.sleep(1)
    # robot.set_digital_output(0, 0, 0)
    # time.sleep(1)

robot.logout()