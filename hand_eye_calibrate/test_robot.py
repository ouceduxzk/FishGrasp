import jkrc 

robot = jkrc.RC("192.168.80.116")
robot.login()
robot.enable_robot()

# robot.linear_move([0,0,0,0,0,0],1,True,20)

# robot.close()
import pdb; pdb.set_trace()
robot.set_digital_output(0, 0, 1)#设置DO2的引脚输出值为1   