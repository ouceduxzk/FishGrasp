 python3 detection/test_yolo.py \
    --weights runs/train/single_yolov8s_20251106_091917/weights/best.pt \
    --source /home/ai/AI_perception/realtime_output/rgb \
    --imgsz 640 \
    --conf 0.5 \
    --mode predict \
    --project runs/predict \
    --name test2
