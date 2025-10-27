 python3 detection/test_yolo.py \
    --weights runs/train/yolov8s_strong_aug/weights/best.pt \
    --source ./realtime_output/failure_case/ \
    --imgsz 640 \
    --conf 0.5 \
    --mode predict \
    --project runs/predict \
    --name test2
