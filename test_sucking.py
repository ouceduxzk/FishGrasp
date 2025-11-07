import time
import jkrc 
import numpy as np 
import json
import argparse
import os
parser = argparse.ArgumentParser(description='Test sucking - load path JSON via outputstr')
parser.add_argument('--grid_params', type=str, default='configs/fish_grid_params.json', help='Path to grid params JSON to read rows/cols')
args = parser.parse_args()

# load rows/cols from grid params (if available)
rows = None
cols = None
fish_paths = None
try:
    if args.grid_params and os.path.exists(args.grid_params):
        with open(args.grid_params, 'r', encoding='utf-8') as f:
            gp = json.load(f)

        fish_paths_path = gp.get('output', {}).get('waypoints_json_path', '')
        grid = gp.get('grid', {})
        rows = int(grid.get('rows', 0))
        cols = int(grid.get('cols', 0))
        print(f"Grid params: rows={rows}, cols={cols}")
        
        # Load the fish paths JSON file
        if fish_paths_path and os.path.exists(fish_paths_path):
            with open(fish_paths_path, 'r', encoding='utf-8') as f:
                fish_paths = json.load(f)
            print(f"Loaded fish paths from: {fish_paths_path}")
            print(f"Number of waypoints: {len(fish_paths)}")
        else:
            print(f"Warning: Fish paths file not found: {fish_paths_path}")
except Exception as e:
    print(f"Failed to load grid params {args.grid_params}: {e}")

robot = jkrc.RC("192.168.80.116")
robot.login()   
robot.power_on()
robot.enable_robot()

debugRobotPath = True

# ret = robot.joint_move([np.pi * 0.9 , 0, 0, 0, 0, 0], 1, True, 1)
ret = robot.joint_move([-204.939*np.pi/180,
                        101.019*np.pi/180, 
                        -62.559*np.pi/180, 
                        53.687*np.pi/180,
                        -269.533*np.pi/180, 
                        -203.005*np.pi/180], 0, False, 0.5)

time.sleep(1)

#robot.set_digital_output(0, 0, 1)
#time.sleep(10)
#robot.set_digital_output(0, 0, 0)

#ret = robot.linear_move([10, -220, -0, 0, 0, 0], 1, True, 100)
#ret = robot.linear_move([0, -100, -50, 0, 0, 0], 1, True, 100)

# Validate that required data is loaded
#if fish_paths is None:
#    print("Error: fish_paths not loaded. Cannot continue.")
#    robot.logout()
#    exit(1)
#
#if rows is None or cols is None or rows <= 0 or cols <= 0:
#    print(f"Error: Invalid grid dimensions (rows={rows}, cols={cols}). Cannot continue.")
#    robot.logout()
#    exit(1)
#
#print(f"Starting to process {rows * cols} waypoints...")
#
#for n in range(rows * cols):
#   account = n + 1
#
#   # get the first point
#   xy_path = fish_paths[str(account)]
#   joint_pos1 = [0, 0, 0, 0, 0, 0]
#   joint_pos1[0] = xy_path[0][0]
#   joint_pos1[1] = xy_path[0][1]
#   joint_pos1[2] = 0
#   joint_pos1[3] = 0
#   joint_pos1[4] = 0
#   joint_pos1[5] = 0
#
#   # move to the first point
#   ret = robot.linear_move(joint_pos1, 1, True, 100)
#
#   # get the second point
#   joint_pos2 = [0, 0, 0, 0, 0, 0]
#   joint_pos2[0] = xy_path[1][0]
#   joint_pos2[1] = xy_path[1][1]
#   joint_pos2[2] = -200
#   joint_pos2[3] = 0
#   joint_pos2[4] = 0
#   joint_pos2[5] = 0
#
#   #move to the second point
#   ret = robot.linear_move(joint_pos2, 1, True, 100)
#
#   time.sleep(2)
#
#   ret = robot.linear_move([0,-joint_pos2[1],200,0,0,0], 1, True, 100)
#
#   ret = robot.joint_move([-204.939*np.pi/180,
#                        101.019*np.pi/180, 
#                        -62.559*np.pi/180, 
#                        53.687*np.pi/180,
#                        -269.533*np.pi/180, 
#                        -203.005*np.pi/180], 0, False, 1)
#
#   time.sleep(1)
#
#   print("times:",n)

robot.logout()