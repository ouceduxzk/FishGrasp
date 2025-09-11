#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultralytics YOLO 测试脚本
- validate 模式：在 dataset.yaml 上评测 mAP、精度/召回
- predict  模式：对图片/文件夹/视频/摄像头进行推理并保存可视化
- test     模式：在指定 source 数据上评估模型指标（mAP、精度、召回等）
- annotation 模式：对图像进行检测并生成LabelMe格式的JSON标注文件（保存到与图像相同目录）

示例：
  # 在验证集上评测
  python3 detection/test_yolo.py \
    --weights runs/train/single_yolov8s_20250908_152257/weights/best.pt \
    --data ../datasets/yolo_dataset/dataset.yaml \
    --imgsz 640 \
    --mode validate

  # 对文件夹做推理并保存结果
  python3 detection/test_yolo.py \
    --weights runs/train/yolov8s_strong_20250909_114704/weights/best.pt \
    --source ./datasets/l0_9.9_sum/images/test \
    --imgsz 640 \
    --conf 0.25 \
    --mode predict \
    --project runs/predict \
    --name demo

  # 在指定数据上评估模型指标
  python3 detection/test_yolo.py \
    --weights runs/train/yolov8s_strong_20250909_114704/weights/best.pt \
    --source ./datasets/l0_9.9_sum/images/test \
    --imgsz 640 \
    --conf 0.25 \
    --mode test \
    --project runs/test \
    --name eval_test

  # 生成LabelMe格式的JSON标注文件，并像 predict 模式一样输出预测可视化
  python3 detection/test_yolo.py \
    --weights runs/train/yolov8s_strong_20250909_114704/weights/best.pt \
    --source ./7009-7509\
    --imgsz 640 \
    --conf 0.25 \
    --mode annotation \
    --project runs/annotations\
    --name annotations14
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO 模型测试/推理")
    ap.add_argument("--weights", required=True, help="权重路径，例如 runs/train/.../weights/best.pt")
    ap.add_argument("--mode", choices=["validate", "predict", "test", "annotation"], default="validate", help="测试模式")
    ap.add_argument("--data", default="", help="dataset.yaml 路径(用于 validate)")
    ap.add_argument("--source", default="", help="推理输入(用于 predict/test)：图片/文件夹/视频/rtsp/0(摄像头)")
    ap.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值(预测)")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IOU 阈值(预测)")
    ap.add_argument("--device", default="", help="CUDA 设备，如 '0' 或 '0,1'，留空自动")
    ap.add_argument("--project", default="runs/test", help="输出根目录")
    ap.add_argument("--name", default="", help="任务名(默认自动)")
    ap.add_argument("--save_txt", action="store_true", help="保存 YOLO txt 结果(预测)")
    ap.add_argument("--save_conf", action="store_true", help="在 txt 中保存置信度")
    ap.add_argument("--recursive", action="store_true", help="当 --source 为目录时递归查找(默认仅当前目录)")
    return ap.parse_args()


def list_images_in_dir(dir_path: Path, recursive: bool = False):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if recursive:
        return [p for p in dir_path.rglob("*") if p.suffix.lower() in exts]
    else:
        return [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]


def evaluate_on_source(model, source_path: str, args):
    """
    在指定的 source 数据上评估模型指标
    """
    from ultralytics.utils.metrics import ConfusionMatrix
    import torch
    import numpy as np
    
    src = Path(source_path)
    if not src.exists():
        print(f"[错误] 源路径不存在: {src}")
        return None
    
    # 收集所有图像文件
    if src.is_dir():
        image_files = list_images_in_dir(src, recursive=args.recursive)
        if not image_files:
            print(f"[错误] 目录下未找到图像: {src}")
            return None
    else:
        image_files = [src]
    
    print(f"[测试] 找到 {len(image_files)} 个图像文件")
    
    # 创建临时 dataset.yaml 用于评估
    import tempfile
    import yaml
    
    # 假设标签文件在对应的 labels 目录中
    labels_dir = src.parent.parent / "labels" / src.name
    if not labels_dir.exists():
        print(f"[警告] 未找到标签目录: {labels_dir}")
        print("[测试] 将仅进行推理，无法计算精确指标")
        return run_inference_only(model, image_files, args)
    
    # 创建临时 dataset.yaml
    temp_yaml = {
        'path': str(src.parent),
        'train': '',  # 不使用
        'val': str(src.relative_to(src.parent)),  # 使用 source 作为验证集
        'test': '',
        'nc': 1,  # 假设单类检测
        'names': ['fish']  # 类别名称
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(temp_yaml, f)
        temp_yaml_path = f.name
    
    try:
        # 使用 model.val 进行评估
        print("[测试] 开始评估...")
        metrics = model.val(
            data=temp_yaml_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device if args.device else None,
            project=args.project,
            name=args.name or "test_eval",
            verbose=True,
            save_json=True,  # 保存详细结果
        )
        
        # 打印详细指标
        print("\n" + "="*50)
        print("评估结果:")
        print("="*50)

        # print number of images
        print(f"总图像数:    {len(image_files)}")
        
        if hasattr(metrics, 'box'):
            box_metrics = metrics.box
            print(f"mAP@0.5:     {box_metrics.map50:.4f}")
            print(f"mAP@0.5:0.95: {box_metrics.map:.4f}")
            print(f"Precision:   {box_metrics.mp:.4f}")
            print(f"Recall:      {box_metrics.mr:.4f}")
            print(f"F1-Score:    {2 * box_metrics.mp * box_metrics.mr / (box_metrics.mp + box_metrics.mr):.4f}")
        
        if hasattr(metrics, 'speed'):
            speed = metrics.speed
            print(f"推理速度:    {speed['inference']:.2f} ms/image")
            print(f"NMS速度:     {speed['postprocess']:.2f} ms/image")
        
        print("="*50)
        
        return metrics
        
    except Exception as e:
        print(f"[错误] 评估失败: {e}")
        print("[测试] 回退到仅推理模式...")
        return run_inference_only(model, image_files, args)
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_yaml_path)
        except:
            pass


def run_inference_only(model, image_files, args):
    """
    仅进行推理，计算基本统计信息
    """
    print("[测试] 仅推理模式 - 无法计算精确指标")
    
    total_detections = 0
    total_images = len(image_files)
    confidences = []
    
    for img_path in image_files:
        try:
            results = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device if args.device else None,
                verbose=False,
                save=True,  # 同步保存Ultralytics自带的可视化结果
                project=args.project,
                name=args.name or "annotations",
            )
            
            if results and len(results) > 0:
                res = results[0]
                if hasattr(res, 'boxes') and res.boxes is not None:
                    num_dets = len(res.boxes)
                    total_detections += num_dets
                    
                    # 收集置信度
                    for box in res.boxes:
                        conf = float(box.conf[0].tolist())
                        confidences.append(conf)
                        
        except Exception as e:
            print(f"[警告] 处理图像失败 {img_path}: {e}")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("推理统计:")
    print("="*50)
    print(f"总图像数:    {total_images}")
    print(f"总检测数:    {total_detections}")
    print(f"平均检测数:  {total_detections/total_images:.2f}")
    
    if confidences:
        confidences = np.array(confidences)
        print(f"平均置信度:  {confidences.mean():.4f}")
        print(f"最高置信度:  {confidences.max():.4f}")
        print(f"最低置信度:  {confidences.min():.4f}")
    
    print("="*50)
    print("[注意] 这是仅推理模式，无法计算 mAP 等精确指标")
    print("       如需精确评估，请确保有对应的标签文件")
    
    return None


def generate_labelme_annotations(model, source_path: str, args):
    """
    对图像进行检测并生成LabelMe格式的JSON标注文件（保存到 runs/<project>/<name>/），
    可视化结果由 Ultralytics 的 predict 保存（与 predict 模式一致）。
    
    Args:
        model: YOLO模型
        source_path: 图像目录路径
        args: 命令行参数
    """
    import json
    import cv2
    import numpy as np
    
    src = Path(source_path)
    if not src.exists():
        print(f"[错误] 源路径不存在: {src}")
        return
    
    # 收集所有图像文件
    if src.is_dir():
        image_files = list_images_in_dir(src, recursive=args.recursive)
        if not image_files:
            print(f"[错误] 目录下未找到图像: {src}")
            return
    else:
        image_files = [src]
    
    print(f"[标注] 找到 {len(image_files)} 个图像文件")
    
    # 创建输出目录 runs/<project>/<name>/
    project_dir = Path(args.project) #/ (args.name or "annotations")
    project_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.project) / (args.name or "annotations")
    
    # 先运行一次批量预测，确保所有可视化结果保存到指定目录
    print(f"[标注] 批量预测并保存可视化结果到: {output_dir}")
    results = model.predict(
        source=[str(p) for p in image_files],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device if args.device else None,
        verbose=False,
        save=True,
        project=args.project,
        name=args.name or "annotations",
    )
    
    processed_count = 0
    failed_count = 0
    
    for idx, img_path in enumerate(image_files):
        try:
            # 获取对应的预测结果
            res = results[idx] if results and idx < len(results) else None
            if res is None:
                print(f"[错误] 缺少预测结果: {img_path}")
                failed_count += 1
                continue
            
            # 获取图像尺寸
            if hasattr(res, 'orig_shape') and res.orig_shape is not None:
                img_height, img_width = int(res.orig_shape[0]), int(res.orig_shape[1])
            else:
                # 回退读取图像
                original_img = cv2.imread(str(img_path))
                if original_img is None:
                    print(f"[错误] 无法读取图像: {img_path}")
                    failed_count += 1
                    continue
                img_height, img_width = original_img.shape[:2]
            
            # 构建LabelMe格式的JSON
            labelme_data = {
                "version": "5.8.3",
                "flags": {},
                "shapes": [],
                "imagePath": img_path.name,
                "imageData": None,
                "imageHeight": img_height,
                "imageWidth": img_width
            }
            
            # 处理检测结果
            if hasattr(res, 'boxes') and res.boxes is not None:
                for i, box in enumerate(res.boxes):
                    # 获取边界框坐标
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0].tolist())
                    cls = int(box.cls[0].tolist())
                    
                    # 创建矩形标注（LabelMe）
                    shape = {
                        "label": "鱿鱼",
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": None,
                        "description": f"confidence: {conf:.3f}",
                        "shape_type": "rectangle",
                        "flags": {},
                        "mask": None
                    }
                    labelme_data["shapes"].append(shape)
            
            # 文件名
            json_filename = img_path.stem + ".json"
            json_path = output_dir / json_filename
            
            # 保存JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, ensure_ascii=False, indent=2)
            
            # 可视化由 Ultralytics 自动保存至同目录
            print(f"[成功] {img_path.name} -> {json_filename} (检测到 {len(labelme_data['shapes'])} 个目标)")
            processed_count += 1
            
        except Exception as e:
            print(f"[错误] 处理图像失败 {img_path.name}: {e}")
            failed_count += 1
    
    # 输出统计信息
    print("\n" + "="*60)
    print("标注生成完成:")
    print("="*60)
    print(f"成功处理: {processed_count} 个图像")
    print(f"处理失败: {failed_count} 个图像")
    print(f"输出目录: {output_dir}")
    print("生成文件:")
    print("  - JSON标注文件: *.json")
    print("  - 预测可视化: 由 Ultralytics 自动保存 (与 predict 模式一致)")
    print("="*60)


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[错误] 未找到 ultralytics，请先: pip install ultralytics")
        print(e)
        sys.exit(1)
    
    # 检查matplotlib兼容性
    try:
        import matplotlib
        import numpy as np
        # 测试matplotlib和numpy的兼容性
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        print(f"[信息] Matplotlib版本: {matplotlib.__version__}")
        print(f"[信息] NumPy版本: {np.__version__}")
    except Exception as e:
        print(f"[警告] Matplotlib/NumPy兼容性问题: {e}")
        print("[建议] 尝试升级matplotlib: pip install matplotlib --upgrade")
        # 继续执行，但可能会在后续遇到问题

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
    elif args.mode == "test":
        if not args.source:
            print("[错误] test 模式需要 --source 输入")
            sys.exit(1)
        metrics = evaluate_on_source(model, args.source, args)
        if metrics:
            print("测试完成。")
        else:
            print("测试完成（仅推理模式）。")
    elif args.mode == "annotation":
        if not args.source:
            print("[错误] annotation 模式需要 --source 输入")
            sys.exit(1)
        generate_labelme_annotations(model, args.source, args)
        print("标注生成完成。")
    else:  # predict mode
        if not args.source:
            print("[错误] predict 模式需要 --source 输入")
            sys.exit(1)

        src = Path(args.source)
        sources = None
        if src.is_dir():
            # 严格限制在该目录（默认非递归）
            sources = list_images_in_dir(src, recursive=args.recursive)
            if not sources:
                print(f"[错误] 目录下未找到图像: {src}")
                sys.exit(1)
        else:
            # 单个文件或其他流式输入，直接传递
            sources = [str(src)]

        results = model.predict(
            source=[str(p) for p in sources],
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
