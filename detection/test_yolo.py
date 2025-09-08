#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultralytics YOLO 测试脚本
- validate 模式：在 dataset.yaml 上评测 mAP、精度/召回
- predict  模式：对图片/文件夹/视频/摄像头进行推理并保存可视化

示例：
  # 在验证集上评测
  python train/test_yolo.py \
    --weights runs/train/your_exp/weights/best.pt \
    --data ../datasets/yolo_dataset/dataset.yaml \
    --imgsz 640 \
    --mode validate

  # 对文件夹做推理并保存结果
  python train/test_yolo.py \
    --weights runs/train/your_exp/weights/best.pt \
    --source ./datasets/yolo_dataset/images/val \
    --imgsz 640 \
    --conf 0.25 \
    --mode predict \
    --project runs/predict \
    --name demo
"""

import os
import sys
import argparse


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO 模型测试/推理")
    ap.add_argument("--weights", required=True, help="权重路径，例如 runs/train/.../weights/best.pt")
    ap.add_argument("--mode", choices=["validate", "predict"], default="validate", help="测试模式")
    ap.add_argument("--data", default="", help="dataset.yaml 路径(用于 validate)")
    ap.add_argument("--source", default="", help="推理输入(用于 predict)：图片/文件夹/视频/rtsp/0(摄像头)")
    ap.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值(预测)")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IOU 阈值(预测)")
    ap.add_argument("--device", default="", help="CUDA 设备，如 '0' 或 '0,1'，留空自动")
    ap.add_argument("--project", default="runs/test", help="输出根目录")
    ap.add_argument("--name", default="", help="任务名(默认自动)")
    ap.add_argument("--save_txt", action="store_true", help="保存 YOLO txt 结果(预测)")
    ap.add_argument("--save_conf", action="store_true", help="在 txt 中保存置信度")
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[错误] 未找到 ultralytics，请先: pip install ultralytics")
        print(e)
        sys.exit(1)

    model = YOLO(args.weights)

    if args.mode == "validate":
        if not args.data:
            print("[错误] validate 模式需要 --data 指向 dataset.yaml")
            sys.exit(1)
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            device=args.device if args.device else None,
            project=args.project,
            name=args.name or "val",
            iou=args.iou,
            verbose=True,
        )
        # metrics 包含各项指标，可按需打印
        print("评测完成。mAP50:", getattr(metrics, 'box', None).map50 if hasattr(metrics, 'box') else None)
    else:
        if not args.source:
            print("[错误] predict 模式需要 --source 输入")
            sys.exit(1)
        results = model.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device if args.device else None,
            project=args.project,
            name=args.name or "pred",
            save=True,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            verbose=True,
        )
        print("推理完成。结果目录:", results[0].save_dir if results else args.project)


if __name__ == "__main__":
    main()
