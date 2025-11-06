import argparse
import os
from pathlib import Path
from typing import List

import cv2


def list_image_files(input_dir: Path, recursive: bool) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if recursive:
        candidates = input_dir.rglob("*")
    else:
        candidates = input_dir.glob("*")
    return [p for p in candidates if p.suffix.lower() in exts and p.is_file()]


def annotate_and_save(image_path: Path, result, output_path: Path) -> None:
    annotated = result.plot()  # numpy BGR image with boxes and scores
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)


def save_txt_detections(result, txt_path: Path) -> None:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return
    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else None
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    cls = boxes.cls.cpu().numpy() if boxes.cls is not None else None
    if xyxy is None or len(xyxy) == 0:
        return
    lines = []
    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = b[:4].tolist()
        s = float(conf[i]) if conf is not None else -1.0
        c = int(cls[i]) if cls is not None else -1
        lines.append(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {s:.4f} {c}")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def run_inference(
    weights: str,
    input_dir: str,
    output_dir: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    recursive: bool = False,
    save_txt: bool = True,
):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from e

    model = YOLO(weights)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(input_path, recursive)
    if not image_files:
        print("No images found to process.")
        return

    for img_file in image_files:
        rel = img_file.relative_to(input_path)
        out_img = output_path / rel
        out_img = out_img.with_suffix(".jpg")
        results = model.predict(
            source=[str(img_file)],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
            save=False,
        )
        if not results:
            continue
        res = results[0]
        annotate_and_save(img_file, res, out_img)
        if save_txt:
            out_txt = out_img.with_suffix(".txt")
            save_txt_detections(res, out_txt)

    print(f"Done. Annotated images saved to: {str(output_path)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference over a folder and save annotated outputs."
    )
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output folder for results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively search input folder"
    )
    parser.add_argument(
        "--no-txt",
        dest="save_txt",
        action="store_false",
        help="Do not save .txt detections alongside images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        weights=args.weights,
        input_dir=args.input,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        recursive=args.recursive,
        save_txt=args.save_txt,
    )


