# YOLO 训练使用说明

本目录提供 Ultralytics YOLO 的训练脚本 `train_yolo.py`。

## 环境准备

- Python 3.8+
- 安装依赖：

```bash
pip install ultralytics
```

## 数据集准备

准备一个数据集配置文件（YAML），例如 `dataset.yaml`：

```yaml
path: /abs/path/to/dataset
train: images/train
val: images/val
# test: images/test   # 可选
names: ["class0", "class1"]
```

目录结构示例：

```
/abs/path/to/dataset
├── images
│   ├── train
│   └── val
├── labels
│   ├── train
│   └── val
└── dataset.yaml
```

- 标注格式：YOLO TXT（每张图片对应一个同名 .txt 文件）
- TXT内容：`class x_center y_center width height`（相对归一化坐标）

## 训练命令示例

```bash
python train_yolo.py \
  --data /abs/path/to/dataset/dataset.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --project runs/train \
  --name fish_yolov8s_$(date +%Y%m%d_%H%M%S) \
  --export
```

参数说明：
- `--data`: 数据集YAML路径
- `--model`: 模型（yolov8n.pt/yolov8s.pt/自定义权重）
- `--epochs`: 训练轮数
- `--batch`: 批大小
- `--imgsz`: 输入尺寸
- `--project`: 输出根目录（默认 `runs/train`）
- `--name`: 实验名称（默认使用模型名+时间戳）
- `--export`: 训练完成后导出 ONNX 和 TorchScript

## 输出

- 训练结果保存在 `runs/train/<name>/`：
  - `weights/best.pt` 最佳权重
  - `weights/last.pt` 最新权重
  - `results.png` 指标曲线
  - `confusion_matrix.png` 混淆矩阵（若有）

## 恢复训练

```bash
python train_yolo.py --data /abs/path/to/dataset.yaml --model runs/train/<name>/weights/last.pt --resume
```
