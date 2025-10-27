#python3 train_landmark_model.py --mode train  --sharpness=3.0      --data_dir ./process_data        --annotations process_data/train_annotations.json        --epochs 100 --backbone=efficientnet --batch_size=128 --lr=0.001 --exp_name gaussian_$(date +%Y%m%d_%H%M%S)


python3 train_landmark_model.py --mode train \
    --data_dir ./process_data \
    --annotations ./process_data/train_annotations.json \
    --epochs 150 \
    --sharpness 1.0 \
    --exp_name ellipsoid_$(date +%Y%m%d_%H%M%S)