#!/usr/bin/env python3
"""
Simple demo: segment a squid (or any object) from an image using Meta's Segment Anything (SAM).

This is NOT a ROS node. It's a plain Python script for quick experimentation.

Usage examples:
  1) GroundingDINO + SAM (recommended - detects objects with text prompt, then segments):
     python3 squid_segment_sam.py \
       --image /path/to/image.jpg \
       --checkpoint /path/to/sam_vit_b_01ec64.pth \
       --model-type vit_b \
       --use-groundingdino \
       --text-prompt "squid"

  2) YOLO + SAM (detects objects first, then segments):
     python3 squid_segment_sam.py \
       --image /path/to/image.jpg \
       --checkpoint /path/to/sam_vit_b_01ec64.pth \
       --model-type vit_b \
       --use-yolo

  3) Automatic (no prompt, picks the largest mask):
     python3 squid_segment_sam.py \
       --image /path/to/image.jpg \
       --checkpoint /path/to/sam_vit_b_01ec64.pth \
       --model-type vit_b

  4) With a point prompt (x,y in image pixels, typically click on the squid):
     python3 squid_segment_sam.py \
       --image /path/to/image.jpg \
       --checkpoint /path/to/sam_vit_b_01ec64.pth \
       --model-type vit_b \
       --point 350,220

Outputs:
  - Saves an overlay PNG highlighting the selected mask
  - Saves a binary mask PNG

Dependencies (install in a virtualenv is recommended):
  - numpy, opencv-python, pillow, matplotlib
  - torch, torchvision (install per your CUDA/CPU from pytorch.org)
  - segment-anything (install via GitHub):
      pip install git+https://github.com/facebookresearch/segment-anything
  - ultralytics (for YOLO detection):
      pip install ultralytics
  - groundingdino-py (for GroundingDINO detection):
      pip install groundingdino-py

Model checkpoints (for 8GB RAM, use vit_b):
  - ViT-B (375MB, recommended): sam_vit_b_01ec64.pth
    Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  - ViT-L (1.2GB): sam_vit_l_0b3195.pth
    Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
  - ViT-H (2.4GB): sam_vit_h_4b8939.pth
    Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception as exc:
    print("PyTorch is required. Please install torch/torchvision as per https://pytorch.org/", file=sys.stderr)
    raise

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError:
    print(
        "Missing dependency 'segment-anything'. Install via:\n"
        "  pip install git+https://github.com/facebookresearch/segment-anything",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "Missing dependency 'ultralytics'. Install via:\n"
        "  pip install ultralytics",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.util.inference import annotate, load_image, predict
except ImportError:
    print(
        "Missing dependency 'groundingdino-py'. Install via:\n"
        "  pip install groundingdino-py",
        file=sys.stderr,
    )
    sys.exit(1)


def load_image_bgr(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image_bgr


def create_overlay(image_bgr: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.6) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    overlay = image_bgr.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)

    # Blend color over masked region
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_arr

    # Draw boundary for visual clarity
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=2)

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def parse_point(point_str: Optional[str]) -> Optional[Tuple[int, int]]:
    if not point_str:
        return None
    try:
        x_str, y_str = point_str.split(",")
        return int(x_str), int(y_str)
    except Exception:
        raise ValueError("--point must be in the form 'x,y' (e.g., 350,220)")


def select_largest_mask(masks: list) -> np.ndarray:
    if not masks:
        raise RuntimeError("No masks generated.")
    # Each entry: { 'segmentation': np.ndarray(H,W,bool), 'area': int, ... }
    best = max(masks, key=lambda m: int(m.get('area', 0)))
    return best['segmentation']


def init_sam(checkpoint_path: str, model_type: str, device: str):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
    if model_type not in sam_model_registry:
        raise ValueError(f"Invalid model_type '{model_type}'. Options: {list(sam_model_registry.keys())}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return sam


def run_with_point_prompt(image_bgr: np.ndarray, point_xy: Tuple[int, int], sam, device: str) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    input_point = np.array([[point_xy[0], point_xy[1]]])  # (x,y)
    input_label = np.array([1], dtype=np.int32)  # 1 indicates foreground

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool)


def detect_objects_with_yolo(image_bgr: np.ndarray, yolo_model: str = "yolov8n.pt", conf_threshold: float = 0.25) -> list:
    """Detect objects using YOLO and return detection results."""
    model = YOLO(yolo_model)
    results = model(image_bgr, conf=conf_threshold, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': class_name,
                    'class_id': cls,
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                })
    
    return detections


def init_groundingdino():
    """Initialize GroundingDINO model."""
    config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "groundingdino_swint_ogc.pth"
    
    # Download config if not exists
    if not os.path.exists(config_file):
        os.makedirs("groundingdino/config", exist_ok=True)
        import urllib.request
        config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        urllib.request.urlretrieve(config_url, config_file)
    
    # Download checkpoint if not exists
    if not os.path.exists(checkpoint_path):
        print("📥 Downloading GroundingDINO checkpoint...")
        import urllib.request
        checkpoint_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
    
    args = SLConfig.fromfile(config_file)
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model


def detect_objects_with_groundingdino(image_bgr: np.ndarray, text_prompt: str, conf_threshold: float = 0.35) -> list:
    """Detect objects using GroundingDINO with text prompt and return detection results."""
    model = init_groundingdino()
    
    # Convert BGR to RGB for GroundingDINO
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for GroundingDINO
    from PIL import Image
    pil_image = Image.fromarray(image_rgb)
    
    # Run GroundingDINO prediction
    detections = predict(
        model=model,
        image=pil_image,
        caption=text_prompt,
        box_threshold=conf_threshold,
        text_threshold=conf_threshold
    )
    
    results = []
    if detections is not None and len(detections) > 0:
        boxes = detections['boxes']
        scores = detections['scores']
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.cpu().numpy()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(score),
                'class': text_prompt,
                'center': [center_x, center_y]
            })
    
    return results


def run_with_groundingdino(image_bgr: np.ndarray, sam, text_prompt: str = "squid", conf_threshold: float = 0.35) -> np.ndarray:
    """Use GroundingDINO to detect objects with text prompt, then use SAM for precise segmentation."""
    print(f"🔍 Detecting objects with GroundingDINO using text prompt: '{text_prompt}'...")
    detections = detect_objects_with_groundingdino(image_bgr, text_prompt, conf_threshold)
    
    if not detections:
        print(f"⚠️  No objects detected by GroundingDINO for '{text_prompt}', falling back to automatic SAM...")
        return run_automatic(image_bgr, sam)
    
    # Sort by confidence and get the best detection
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    best_detection = detections[0]
    
    print(f"🎯 Best detection: {best_detection['class']} (confidence: {best_detection['confidence']:.2f})")
    print(f"📍 Bounding box: {best_detection['bbox']}")
    print(f"🎯 Center point: {best_detection['center']}")
    
    # Use the center of the bounding box as the point prompt for SAM
    center_point = best_detection['center']
    return run_with_point_prompt(image_bgr, center_point, sam, "cpu")


def run_with_yolo_detection(image_bgr: np.ndarray, sam, yolo_model: str = "yolov8n.pt", conf_threshold: float = 0.25) -> np.ndarray:
    """Use YOLO to detect objects, then use SAM for precise segmentation."""
    print("🔍 Detecting objects with YOLO...")
    detections = detect_objects_with_yolo(image_bgr, yolo_model, conf_threshold)
    
    if not detections:
        print("⚠️  No objects detected by YOLO, falling back to automatic SAM...")
        return run_automatic(image_bgr, sam)
    
    # Sort by confidence and get the best detection
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    best_detection = detections[0]
    
    print(f"🎯 Best detection: {best_detection['class']} (confidence: {best_detection['confidence']:.2f})")
    print(f"📍 Bounding box: {best_detection['bbox']}")
    print(f"🎯 Center point: {best_detection['center']}")
    
    # Use the center of the bounding box as the point prompt for SAM
    center_point = best_detection['center']
    return run_with_point_prompt(image_bgr, center_point, sam, "cpu")


def run_automatic(image_bgr: np.ndarray, sam) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=50,
    )
    masks = mask_generator.generate(image_rgb)
    return select_largest_mask(masks)


def main():
    parser = argparse.ArgumentParser(description="Segment a squid (or any object) from an image using SAM")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", "-c", default=os.environ.get("SAM_CHECKPOINT", ""), help="Path to SAM model checkpoint (e.g., sam_vit_b_01ec64.pth for 8GB RAM)")
    parser.add_argument("--model-type", "-m", default="vit_b", help="SAM model type: vit_h | vit_l | vit_b (vit_b recommended for 8GB RAM)")
    parser.add_argument("--device", "-d", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device: cuda or cpu")
    parser.add_argument("--point", "-p", default=None, help="Optional point prompt as 'x,y' (e.g., 350,220)")
    parser.add_argument("--output", "-o", default=None, help="Output basename (without extension). If omitted, uses image filename stem.")
    parser.add_argument("--use-yolo", "-y", action="store_true", help="Use YOLO for object detection first, then SAM for segmentation")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model to use (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold (0.0-1.0)")
    parser.add_argument("--use-groundingdino", action="store_true", help="Use GroundingDINO for object detection with a text prompt")
    parser.add_argument("--text-prompt", default="squid", help="Text prompt for GroundingDINO (e.g., 'squid', 'octopus', 'fish')")

    args = parser.parse_args()

    if not args.checkpoint:
        print("--checkpoint is required (or set SAM_CHECKPOINT env var)", file=sys.stderr)
        sys.exit(2)

    image_bgr = load_image_bgr(args.image)
    sam = init_sam(args.checkpoint, args.model_type, args.device)

    point_xy = parse_point(args.point)
    if args.use_groundingdino:
        print("🚀 Using GroundingDINO + SAM pipeline...")
        mask = run_with_groundingdino(image_bgr, sam, args.text_prompt)
    elif args.use_yolo:
        print("🚀 Using YOLO + SAM pipeline...")
        mask = run_with_yolo_detection(image_bgr, sam, args.yolo_model, args.yolo_conf)
    elif point_xy is not None:
        print("🎯 Using point prompt...")
        mask = run_with_point_prompt(image_bgr, point_xy, sam, args.device)
    else:
        print("🔍 Using automatic SAM...")
        mask = run_automatic(image_bgr, sam)

    stem = args.output
    if stem is None:
        stem = os.path.splitext(os.path.basename(args.image))[0] + "_sam"
    os.makedirs(os.path.dirname(args.output) if args.output and os.path.dirname(args.output) else ".", exist_ok=True)

    overlay = create_overlay(image_bgr, mask, color=(0, 255, 0), alpha=0.6)
    overlay_path = f"{stem}_overlay.png"
    mask_path = f"{stem}_mask.png"

    # Save overlay (BGR) and binary mask
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(mask_path, (mask.astype(np.uint8) * 255))

    print(f"Saved: {overlay_path}")
    print(f"Saved: {mask_path}")


if __name__ == "__main__":
    main()


