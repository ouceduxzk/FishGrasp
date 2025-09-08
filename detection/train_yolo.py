#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultralytics YOLO 训练脚本
- 加载数据集(YAML)
- 训练并保存结果到本地（按项目/名称分组，名称默认带时间戳）
- 可选导出ONNX与TorchScript

示例：
  python train_yolo.py \
    --data /path/to/dataset.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --project runs/train \
    --name fish_yolov8s_$(date +%Y%m%d_%H%M%S)

数据集YAML示例(data.yaml)：
  path: /abs/path/to/dataset
  train: images/train
  val: images/val
  test: images/test  # 可选
  names: ["class0", "class1", ...]

依赖：
  pip install ultralytics
"""

import os
import sys
import argparse
from datetime import datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO 训练脚本",
    )
    parser.add_argument("--data", type=str, required=True, help="数据集YAML路径，如: /data/dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="模型权重或架构，如 yolov8n.pt/yolov8s.pt 或自定义.pt")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--device", type=str, default="", help="CUDA设备，如 '0' 或 '0,1'，留空自动选择")
    parser.add_argument("--project", type=str, default="runs/train", help="输出项目目录")
    parser.add_argument("--name", type=str, default="", help="实验名称，默认自动加时间戳")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心轮数")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--pretrained", action="store_true", help="是否使用预训练权重")
    parser.add_argument("--cache", action="store_true", help="缓存图像以加速训练")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖已存在的目录")
    parser.add_argument("--resume", action="store_true", help="从最近的断点恢复训练")
    parser.add_argument("--export", action="store_true", help="训练完成后导出ONNX与TorchScript")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[错误] 未找到ultralytics，请先安装: pip install ultralytics")
        print(f"详细错误: {e}")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"[错误] 数据集YAML不存在: {args.data}")
        sys.exit(1)

    # 生成良好命名：若未指定name，使用模型名+时间戳
    if not args.name:
        model_stem = os.path.splitext(os.path.basename(args.model))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"{model_stem}_{timestamp}"

    os.makedirs(args.project, exist_ok=True)

    print("======== 训练参数 ========")
    print(f"data      : {args.data}")
    print(f"model     : {args.model}")
    print(f"epochs    : {args.epochs}")
    print(f"batch     : {args.batch}")
    print(f"imgsz     : {args.imgsz}")
    print(f"device    : {args.device or 'auto'}")
    print(f"project   : {args.project}")
    print(f"name      : {args.name}")
    print(f"workers   : {args.workers}")
    print(f"patience  : {args.patience}")
    print(f"lr0       : {args.lr0}")
    print(f"pretrained: {args.pretrained}")
    print(f"cache     : {args.cache}")
    print(f"seed      : {args.seed}")
    print(f"exist_ok  : {args.exist_ok}")
    print(f"resume    : {args.resume}")
    print("==========================")

    # 创建与加载模型
    model = YOLO(args.model)

    # 训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        cache=args.cache,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        seed=args.seed,
        exist_ok=args.exist_ok,
        resume=args.resume,
        verbose=True,
    )

    # 结果目录与best.pt
    save_dir = os.path.join(args.project, args.name)
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    last_pt = os.path.join(save_dir, "weights", "last.pt")

    print("\n======== 训练完成 ========")
    print(f"保存目录: {save_dir}")
    if os.path.exists(best_pt):
        print(f"最佳权重: {best_pt}")
    if os.path.exists(last_pt):
        print(f"最新权重: {last_pt}")

    # 可选导出
    if args.export and os.path.exists(best_pt):
        try:
            print("\n开始导出ONNX与TorchScript ...")
            # 重新加载最佳权重再导出
            exp_model = YOLO(best_pt)
            onnx_path = exp_model.export(format="onnx", imgsz=args.imgsz)
            torchscript_path = exp_model.export(format="torchscript", imgsz=args.imgsz)
            print(f"ONNX导出: {onnx_path}")
            print(f"TorchScript导出: {torchscript_path}")
        except Exception as e:
            print(f"[警告] 导出失败: {e}")


if __name__ == "__main__":
    main()
