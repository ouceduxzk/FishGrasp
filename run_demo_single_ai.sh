python3 realtime_segmentation_3d.py   --output_dir realtime_output   --device cuda  \
   --use_yolo   --yolo_weights runs/train/single_yolov8s_20250912_134851/weights/best.pt \
   --grasp_point_mode ai \
   --landmark_model_path landmarks/experiments/gaussian_20250916_141345_20250916_141347/best_fish_landmark_model_gaussian.pth \
   --debug
