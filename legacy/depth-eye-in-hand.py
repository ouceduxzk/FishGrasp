import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(grey_frame, None)

img_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))
cv2.imshow("Keypoints", img_keypoints)


R_list = []
t_list = []

R, T = cv2.calibrateHandEye(R_list, t_list)






