import jkrc 
robot = jkrc.RC("192.168.80.116")
robot.login()   
import time 
while True:
    robot.set_digital_output(0, 0, 1)
    time.sleep(1)
    robot.set_digital_output(0, 0, 0)
    time.sleep(1)

robot.logout()