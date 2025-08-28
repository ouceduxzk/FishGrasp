import cv2
import os


for file in os.listdir('pic'):
    if file.endswith('.png'):
        img = cv2.imread(os.path.join('pic', file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)