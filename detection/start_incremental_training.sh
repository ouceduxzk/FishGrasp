#!/bin/bash

# 增量训练启动脚本
# 使用方法: ./start_incremental_training.sh

echo "=== 增量训练启动脚本 ==="

# 设置默认参数
DATA_DIR="./datasets/incremental"
MODEL_PATH="runs/train/yolov8s_strong_20250909_114704/weights/best.pt"
BATCH_SIZE=500
EPOCHS=50
IMGSZ=640
DEVICE="0"
PROJECT="runs/incremental_train"
CHECK_INTERVAL=300

# 检查参数
if [ $# -ge 1 ]; then
    DATA_DIR=$1
fi

if [ $# -ge 2 ]; then
    MODEL_PATH=$2
fi

echo "数据目录: $DATA_DIR"
echo "模型路径: $MODEL_PATH"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "图像尺寸: $IMGSZ"
echo "设备: $DEVICE"
echo "项目目录: $PROJECT"
echo "检查间隔: ${CHECK_INTERVAL}秒"
echo ""

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo "创建数据目录: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "请确保模型文件存在或修改脚本中的MODEL_PATH"
    exit 1
fi

# 启动增量训练
echo "启动增量训练..."
python3 detection/incremental_train.py \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --imgsz $IMGSZ \
    --device "$DEVICE" \
    --project "$PROJECT" \
    --check_interval $CHECK_INTERVAL

echo "增量训练已停止"


