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
                        168.094*np.pi/180], 0, False, 1)

time.sleep(1)

# ret = robot.linear_move([90, -416, 0, 0, 0, 0], 1, True, 50)

# ret = robot.linear_move([0, -54, -100, 0, 0, 0], 1, True, 50)

#robot.motion_abort()

    # time.sleep(1)
    # robot.set_digital_output(0, 0, 0)
    # time.sleep(1)

robot.logout()