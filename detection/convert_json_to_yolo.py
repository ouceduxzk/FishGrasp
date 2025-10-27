#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将包含矩形两点(左上/右下)的 JSON 标注转换为 YOLO TXT 标注。
- 适配示例结构：
  {
    "imagePath": "xxx.jpg",            # 可选
    "imageWidth": 3840,                 # 必需其一(Width/Height或可从图像读)
    "imageHeight": 2160,
    "shapes": [
      {
        "label": "鱿鱼",
        "shape_type": "rectangle",
        "points": [[x1,y1],[x2,y2]]
      },
      ...
    ]
  }
- 归一化为 YOLO: class x_center y_center w h
- 递归遍历 --source 目录下所有 .json，生成与图片同名的 .txt
- 自动收集类别，生成 dataset.yaml（可选指定类别顺序文件）

用法：
  python train/convert_json_to_yolo.py \
    --source ./data \
    --images_subdir images \
    --labels_out ./datasets/yolo_dataset/labels \
    --images_out ./datasets/yolo_dataset/images \
    --copy_images \
    --names_out ./datasets/yolo_dataset/dataset.yaml
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="将两点矩形JSON转换为YOLO TXT")
    ap.add_argument("--source", required=True, help="源根目录(递归查找json)")
    ap.add_argument("--labels_out", required=True, help="输出labels根目录，内部会按train/val/test或保持平铺")
    ap.add_argument("--images_out", default="", help="可选：复制/硬链图片到该images根目录")
    ap.add_argument("--copy_images", action="store_true", help="复制图片到images_out，否则硬链接")
    ap.add_argument("--split", choices=["none","mirror"], default="mirror", help="输出子结构：none=不分层，mirror=镜像源目录层级")
    ap.add_argument("--names_out", default="", help="输出dataset.yaml(若指定且提供images_out)" )
    ap.add_argument("--class_order", default="", help="可选：提供一个txt，按行给定类别顺序")
    return ap.parse_args()


def load_class_order(path: str) -> List[str]:
    if not path or not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def norm_bbox_to_yolo(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float,float,float,float]:
    x_min, y_min = min(x1,x2), min(y1,y2)
    x_max, y_max = max(x1,x2), max(y1,y2)
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    cx = x_min + w/2.0
    cy = y_min + h/2.0
    if W <= 0 or H <= 0:
        raise ValueError("Invalid image size")
    return cx/W, cy/H, w/W, h/H


def find_image_for_json(jpath: Path, image_path_in_json: Optional[str], source_root: Path) -> Optional[Path]:
    # 优先使用 imagePath 相对 json 所在目录
    if image_path_in_json:
        p = (jpath.parent / image_path_in_json).resolve()
        if p.exists() and p.suffix.lower() in IMG_EXTS:
            return p
        # 尝试仅用文件名
        p2 = next((q for q in jpath.parent.glob(Path(image_path_in_json).name) if q.suffix.lower() in IMG_EXTS), None)
        if p2:
            return p2
        # 在整个source_root内按文件名搜索
        cands = list(source_root.rglob(Path(image_path_in_json).name))
        cands = [c for c in cands if c.suffix.lower() in IMG_EXTS]
        if cands:
            return cands[0]
    # 若无imagePath，尝试同名不同后缀
    stem = jpath.stem
    for ext in IMG_EXTS:
        p = (jpath.parent / f"{stem}{ext}")
        if p.exists():
            return p
    # 递归在同目录查找前缀匹配
    cands = list(jpath.parent.glob(f"{jpath.stem}.*"))
    for c in cands:
        if c.suffix.lower() in IMG_EXTS:
            return c
    return None


def convert_one(json_path: Path, labels_root: Path, images_root: Optional[Path], copy_images: bool, split: str, class_order: List[str], seen_classes: Dict[str,int]) -> Optional[str]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"[跳过] 读取失败 {json_path}: {e}"

    imgW = int(data.get("imageWidth", 0) or 0)
    imgH = int(data.get("imageHeight", 0) or 0)
    imagePath_in = data.get("imagePath", "")

    src_img = find_image_for_json(json_path, imagePath_in, json_path.parents[2] if len(json_path.parents)>=2 else json_path.parent)

    # 若无宽高，尝试从图像读
    if (imgW <= 0 or imgH <= 0) and src_img and src_img.exists():
        try:
            from PIL import Image
            with Image.open(src_img) as im:
                imgW, imgH = im.size
        except Exception:
            pass
    if imgW <= 0 or imgH <= 0:
        return f"[跳过] 无法获取尺寸 {json_path}"

    shapes = data.get("shapes") or data.get("annotations") or []
    if not isinstance(shapes, list):
        return f"[跳过] 无有效shapes {json_path}"

    # 目标labels路径
    if split == "mirror":
        rel_dir = json_path.parent.relative_to(Path(args.source).resolve())  # type: ignore[name-defined]
        lbl_dir = labels_root / rel_dir
    else:
        lbl_dir = labels_root

    # 图片输出
    dst_img = None
    if images_root is not None and src_img is not None and src_img.exists():
        if split == "mirror":
            img_rel_dir = json_path.parent.relative_to(Path(args.source).resolve())  # type: ignore[name-defined]
            dst_img = images_root / img_rel_dir / src_img.name
        else:
            dst_img = images_root / src_img.name
        ensure_parent(dst_img)
        try:
            if copy_images:
                shutil.copy2(src_img, dst_img)
            else:
                if dst_img.exists():
                    dst_img.unlink()
                os.link(src_img, dst_img)
        except Exception:
            shutil.copy2(src_img, dst_img)

    # 组装yolo行
    yolo_lines: List[str] = []
    for shp in shapes:
        if shp.get("shape_type") not in (None, "rectangle", "bbox", "box"):
            continue
        pts = shp.get("points")
        if not pts or len(pts) < 2:
            continue
        (x1,y1),(x2,y2) = pts[0], pts[1]
        try:
            xc,yc,w,h = norm_bbox_to_yolo(float(x1),float(y1),float(x2),float(y2), imgW, imgH)
        except Exception:
            continue
        label = shp.get("label", "object")
        if class_order:
            if label not in class_order:
                # 未知类别丢弃或追加？此处选择丢弃
                continue
            cid = class_order.index(label)
        else:
            if label not in seen_classes:
                seen_classes[label] = len(seen_classes)
            cid = seen_classes[label]
        yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # 写出txt
    lbl_path = (lbl_dir / f"{(src_img.stem if src_img else json_path.stem)}.txt")
    ensure_parent(lbl_path)
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    return None


def write_dataset_yaml(images_root: Path, names_out: Path, class_names: List[str]):
    ensure_parent(names_out)
    yaml = (
        f"path: {images_root.parent.resolve()}\n"
        f"train: images\n"
        f"val: images\n"
        f"test: images\n"
        f"names: {class_names}\n"
    )
    with open(names_out, "w", encoding="utf-8") as f:
        f.write(yaml)


def main():
    global args  # 供 convert_one 使用 source 路径
    args = parse_args()

    source = Path(args.source).resolve()
    labels_root = Path(args.labels_out).resolve()
    images_root = Path(args.images_out).resolve() if args.images_out else None

    class_order = load_class_order(args.class_order)
    seen_classes: Dict[str,int] = {}

    json_files = list(source.rglob("*.json"))
    if not json_files:
        print(f"[错误] 未找到json: {source}")
        sys.exit(1)

    errors = 0
    for jp in json_files:
        err = convert_one(jp, labels_root, images_root, args.copy_images, args.split, class_order, seen_classes)
        if err:
            errors += 1
            if errors <= 20:
                print(err)
    print(f"完成。转换json: {len(json_files)}, 错误: {errors}, 类别数: {len(seen_classes) or len(class_order)}")

    # 生成dataset.yaml（仅当提供images_out时）
    if args.names_out and (args.images_out):
        names = class_order if class_order else [None]*len(seen_classes)
        if not class_order:
            # 根据seen_classes的索引顺序重建
            names = [None]*len(seen_classes)
            for name, idx in seen_classes.items():
                names[idx] = name
        write_dataset_yaml(Path(args.images_out), Path(args.names_out), names)  # type: ignore[arg-type]
        print(f"[OK] 写出dataset.yaml: {args.names_out}")


if __name__ == "__main__":
    main()
