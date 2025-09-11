import jkrc 
import numpy as np 
robot = jkrc.RC("192.168.80.116")
robot.login()   
robot.power_on()
robot.enable_robot()

# ret = robot.joint_move([np.pi * 0.9 , 0, 0, 0, 0, 0], 1, True, 1)
#ret = robot.joint_move([np.pi * 0.9 , 0, 0, 0, 0, 0], 1, True, 1)
    # time.sleep(1)
    # robot.set_digital_output(0, 0, 0)
    # time.sleep(1)

robot.logout()