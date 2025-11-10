 python3 detection/test_yolo.py \
    --weights runs/train/single_yolov8s_20251108_163141/weights/best.pt \
    --source /home/ai/AI_perception/datasets/l0_11.04_yolo/images/test \
    --imgsz 640 \
    --conf 0.3 \
    --mode predict \
    --project runs/predict \
    --name test2
