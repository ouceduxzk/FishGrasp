#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO 数据预处理脚本
- 从 source_dir (默认: ../data) 自动发现 COCO 格式的 JSON 标注
- 按比例切分 train/val/test
- 转换为 YOLO TXT 标注，并整理为 Ultralytics YOLO 所需目录结构
- 生成 dataset.yaml 便于训练

用法示例：
  python prepare_yolo_dataset.py \
    --source_dir ../data \
    --out_dir ../datasets/yolo_dataset \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
    --copy

注意：
- 默认假设 JSON 为 COCO 格式（images / annotations / categories）
- 如果同目录存在多个 JSON，会选择包含 images 数量最多的一个
- 支持拷贝(--copy)或硬链接(--link，默认)方式整理图片
"""

import os
import sys
import json
import argparse
import random
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import unquote

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备YOLO数据集(从COCO JSON)")
    parser.add_argument("--source_dir", type=str, default="../data", help="原始数据根目录（包含图像与json）")
    parser.add_argument("--json", type=str, default="", help="指定COCO json路径（可选，默认自动发现）")
    parser.add_argument("--out_dir", type=str, default="../datasets/yolo_dataset", help="输出数据集根目录")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--copy", action="store_true", help="复制图片到目标目录（默认使用硬链接）")
    group.add_argument("--symlink", action="store_true", help="使用软链接")
    return parser.parse_args()


def find_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


def auto_find_json(source_dir: Path) -> Path:
    jsons = list(source_dir.rglob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"未在 {source_dir} 下找到任何json标注文件")
    # 优先选择images数量最多的coco json
    best = None
    best_n = -1
    for j in jsons:
        try:
            with open(j, "r", encoding="utf-8") as f:
                data = json.load(f)
            n = len(data.get("images", []))
            if n > best_n:
                best, best_n = j, n
        except Exception:
            continue
    if best is None:
        raise RuntimeError("未找到可用的COCO json (无images字段)")
    return best


def load_coco(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco.get("images", [])}
    categories = {cat["id"]: cat for cat in coco.get("categories", [])}
    # 将annotations按照image_id聚合
    anns_by_img: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        anns_by_img.setdefault(img_id, []).append(ann)
    return coco, images, anns_by_img, categories


def bbox_coco_to_yolo(bbox: List[float], img_w: int, img_h: int):
    # COCO: [x_min, y_min, w, h] (像素)
    x_min, y_min, w, h = bbox
    x_c = x_min + w / 2.0
    y_c = y_min + h / 2.0
    return x_c / img_w, y_c / img_h, w / img_w, h / img_h


def ensure_dirs(root: Path):
    for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def place_image(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    else:  # hardlink
        if dst.exists():
            dst.unlink()
        # 对跨设备的硬链接失败回退为复制
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def write_label(label_path: Path, lines: List[str]):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def resolve_image_path(source_dir: Path, file_name: str) -> Optional[Path]:
    """尽最大可能解析COCO中的file_name为实际图片路径。"""
    candidates = sanitize_file_name(file_name)
    # 1) 直接拼接相对/绝对路径
    for c in candidates:
        p = (source_dir / c).resolve() if not os.path.isabs(c) else Path(c)
        if p.exists() and p.suffix.lower() in IMG_EXTS:
            return p
    # 2) 在source_dir下用basename递归查找
    base_names = [Path(c).name for c in candidates]
    base_names = list(dict.fromkeys([b for b in base_names if b]))
    for b in base_names:
        found = list(source_dir.rglob(b))
        for p in found:
            if p.suffix.lower() in IMG_EXTS:
                return p
    # 3) 用stem + 扩展名的通配符在全目录搜索
    for c in candidates:
        stem = Path(c).stem
        if not stem:
            continue
        for ext in IMG_EXTS:
            found = list(source_dir.rglob(f"{stem}{ext}"))
            if found:
                return found[0]
    return None


def generate_yaml(out_dir: Path, names: List[str]):
    yaml_path = out_dir / "dataset.yaml"
    content = (
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names: {names}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] 生成数据集配置: {yaml_path}")


def main():
    args = parse_args()
    random.seed(args.seed)

    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    json_path = Path(args.json).resolve() if args.json else None

    if not source_dir.exists():
        print(f"[错误] source_dir 不存在: {source_dir}")
        sys.exit(1)

    # 自动发现JSON
    if json_path is None:
        try:
            json_path = auto_find_json(source_dir)
        except Exception as e:
            print(f"[错误] 自动发现json失败: {e}")
            sys.exit(1)

    print(f"使用标注: {json_path}")

    # 加载COCO
    coco, images, anns_by_img, categories = load_coco(json_path)

    # 类别名列表（按cat_id升序映射到从0开始的class_id）
    cat_id_list = sorted(categories.keys())
    catid2cls = {cat_id: idx for idx, cat_id in enumerate(cat_id_list)}
    names = [categories[cid]["name"] for cid in cat_id_list]

    # 收集有效图像
    valid: List[Tuple[int, Path]] = []
    miss_count = 0
    for img in images.values():
        fn = img.get("file_name", "")
        p = resolve_image_path(source_dir, fn)
        if p is None:
            miss_count += 1
            if miss_count <= 20:
                print(f"[警告] 未找到文件: '{fn}' (将跳过)")
            continue
        valid.append((img["id"], p))

    if not valid:
        print("[错误] 未找到任何有效图像文件，请检查 file_name 与路径/扩展名/编码")
        sys.exit(1)

    # 切分
    img_ids = [img_id for img_id, _ in valid]
    random.shuffle(img_ids)
    n = len(img_ids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val
    train_ids = set(img_ids[:n_train])
    val_ids = set(img_ids[n_train:n_train + n_val])
    test_ids = set(img_ids[n_train + n_val:])

    print(f"总图像: {n} | train: {len(train_ids)} val: {len(val_ids)} test: {len(test_ids)} | 缺失: {miss_count}")

    # 准备目录
    ensure_dirs(out_dir)
    place_mode = "copy" if args.copy else ("symlink" if args.symlink else "hardlink")

    # 处理每张图片与标注
    for img_id, img_path in valid:
        img_info = images[img_id]
        img_w = int(img_info.get("width", 0) or 0)
        img_h = int(img_info.get("height", 0) or 0)
        if img_w <= 0 or img_h <= 0:
            # 若缺失，尝试用PIL读取尺寸
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                print(f"[警告] 无法获取尺寸，跳过: {img_path}")
                continue

        # 生成YOLO标签行
        yolo_lines: List[str] = []
        for ann in anns_by_img.get(img_id, []):
            cat_id = ann.get("category_id")
            if cat_id not in catid2cls:
                continue
            cls_id = catid2cls[cat_id]
            bbox = ann.get("bbox", None)
            if not bbox:
                continue
            x_c, y_c, w, h = bbox_coco_to_yolo(bbox, img_w, img_h)
            # 过滤异常框
            if w <= 0 or h <= 0 or x_c <= 0 or y_c <= 0 or x_c >= 1 or y_c >= 1:
                continue
            yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # 目标子目录
        if img_id in train_ids:
            split = "train"
        elif img_id in val_ids:
            split = "val"
        else:
            split = "test"

        dst_img = out_dir / f"images/{split}/{img_path.name}"
        dst_lbl = out_dir / f"labels/{split}/{img_path.with_suffix('.txt').name}"

        # 放置图片
        try:
            place_image(img_path, dst_img, place_mode)
        except Exception as e:
            print(f"[警告] 放置图片失败 {img_path} -> {dst_img}: {e}")
            continue

        # 写入标签
        write_label(dst_lbl, yolo_lines)

    # 写 dataset.yaml
    generate_yaml(out_dir, names)

    print("[完成] 数据集已整理为YOLO格式")


if __name__ == "__main__":
    main()
