# python3 realtime_segmentation_3d.py   --output_dir realtime_output   --device cuda  \
#    --use_yolo   --yolo_weights runs/train/single_yolov8s_20251106_091917/weights/best.pt \
#    --grasp_point_mode centroid --debug \
#    --landmark_model_path landmarks/experiments/gaussian_20250922_153626_20250922_153630/best_fish_landmark_model_gaussian.pth 


python3 realtime_segmentation_3d.py   --output_dir realtime_output   --device cuda  \
   --use_yolo   --yolo_weights runs/train/single_yolov8s_20250912_134851/weights/best.pt \
   --grasp_point_mode centroid --debug \
   --landmark_model_path landmarks/experiments/gaussian_20250922_153626_20250922_153630/best_fish_landmark_model_gaussian.pth 

