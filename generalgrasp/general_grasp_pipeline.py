#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GeneralGraspPipeline

New class that orchestrates the general grasp flow by CALLING existing
components in the repo (no core logic rewritten):
 - RealSense capture: setup_realsense
 - Models: seg.init_models (GroundingDINO + SAM predictor)
 - Mask→3D mapping: mask_to_3d_pointcloud
 - YOLO detection: ultralytics.YOLO
 - Registration: Open3D (coarse centering + ICP)
 - Pose transform: map known CAD-frame grasp pose to camera/gripper frame
"""

import os
import sys
import time
import numpy as np
import cv2
import open3d as o3d

# Import repo modules (reuse only)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from seg import init_models
from mask_to_3d import mask_to_3d_pointcloud, load_camera_intrinsics
from realsense_capture import setup_realsense


class GeneralGraspPipeline:
    def __init__(self, device: str = "cuda", intrinsics_file: str = None, hand_eye_file: str = None,
                 yolo_weights: str = None, debug: bool = False,
                 det_gray: bool = False, yolo_weights_gray: str = None):
        self.device = device
        self.debug = debug
        self.yolo_weights = yolo_weights
        self.det_gray = det_gray
        self.yolo_weights_gray = yolo_weights_gray

        # Models (GroundingDINO + SAM)
        self.sam_predictor, self.grounding_dino_model, self.processor = init_models(device)

        # Camera
        self.pipeline, _ = setup_realsense()
        if self.pipeline is None:
            raise RuntimeError("无法启动RealSense相机")

        import pyrealsense2 as rs
        self.align = rs.align(rs.stream.color)

        # Intrinsics / hand-eye
        self.fx, self.fy, self.cx, self.cy, self.dist, self.mtx = load_camera_intrinsics(intrinsics_file)
        self.hand_eye_transform = None
        if hand_eye_file and os.path.exists(hand_eye_file):
            try:
                mat = np.load(hand_eye_file)
                if mat.shape == (4, 4):
                    self.hand_eye_transform = mat.astype(np.float32)
            except Exception:
                pass

    def capture_frame(self, timeout_ms: int = 10000):
        import pyrealsense2 as rs
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None, False
            color_image = np.asanyarray(color_frame.get_data())
            h, w = depth_frame.get_height(), depth_frame.get_width()
            depth_image = np.zeros((h, w), dtype=np.uint16)
            for y in range(h):
                for x in range(w):
                    dist = depth_frame.get_distance(x, y)
                    if dist > 0:
                        depth_image[y, x] = int(dist * 1000)
            return color_image, depth_image, True
        except rs.error:
            return None, None, False
        except Exception:
            return None, None, False

    def detect_yolo_all(self, color_image, conf=0.4, iou=0.45, imgsz=640):
        if not self.yolo_weights or not os.path.exists(self.yolo_weights):
            return []
        try:
            from ultralytics import YOLO
        except Exception:
            return []
        try:
            model = YOLO(self.yolo_weights)
            results = model.predict(
                source=[color_image], imgsz=imgsz, conf=conf, iou=iou,
                device=(0 if self.device == 'cuda' else 'cpu'), verbose=False, save=False
            )
        except Exception:
            return []
        if not results:
            return []
        res = results[0]
        if not hasattr(res, 'boxes') or res.boxes is None or res.boxes.shape[0] == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf_arr = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, 'conf') else None
        cls_arr = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, 'cls') else None
        out = []
        for i, b in enumerate(xyxy):
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4].tolist()]
            conf_v = float(conf_arr[i]) if conf_arr is not None else 0.0
            cls_v = int(cls_arr[i]) if cls_arr is not None else -1
            out.append((x1, y1, x2, y2, conf_v, cls_v))
        return out

    def segment_with_sam(self, color_image, bbox):
        x1, y1, x2, y2 = bbox[:4]
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        import torch
        boxes_tensor = torch.tensor([[x1, y1, x2, y2]], device=self.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_tensor, image_rgb.shape[:2])
        masks, scores, logits = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        if masks.shape[0] == 0 or masks.shape[1] == 0:
            return None
        mask_np = masks[0][0].detach().cpu().numpy().astype(np.uint8) * 255
        restricted = np.zeros_like(mask_np, dtype=np.uint8)
        restricted[y1:y2, x1:x2] = mask_np[y1:y2, x1:x2]
        return restricted

    def mask_to_points_cam(self, color_image, depth_image, mask_np):
        depth_m = depth_image.astype(np.float32) / 1000.0
        color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        points, colors = mask_to_3d_pointcloud(
            color_rgb, depth_m, (mask_np > 0),
            self.fx, self.fy, self.cx, self.cy, self.mtx, self.dist
        )
        return points, colors

    def _center_pcd(self, pcd: o3d.geometry.PointCloud):
        pts = np.asarray(pcd.points)
        centroid = pts.mean(axis=0)
        pcd_centered = pcd.translate(-centroid)
        return pcd_centered, centroid

    def register_partial_to_cad(self, points_cam, cad_pcd: o3d.geometry.PointCloud, voxel_size=0.003):
        part = o3d.geometry.PointCloud()
        part.points = o3d.utility.Vector3dVector(points_cam.astype(np.float64))
        part_centered, part_centroid = self._center_pcd(part)
        cad_centered, cad_centroid = self._center_pcd(cad_pcd)

        # Downsample + normals
        src = part_centered.voxel_down_sample(voxel_size)
        tgt = cad_centered.voxel_down_sample(voxel_size)
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))

        threshold = voxel_size * 4
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        T_part_from_cam = np.eye(4, dtype=np.float32)
        T_part_from_cam[:3, 3] = part_centroid.astype(np.float32)
        T_center = reg.transformation.astype(np.float32)
        T_model_from_centered = np.eye(4, dtype=np.float32)
        T_model_from_centered[:3, 3] = cad_centroid.astype(np.float32)
        T_cam_from_part = np.linalg.inv(T_part_from_cam)
        T_model_from_cam = T_model_from_centered @ T_center @ T_cam_from_part
        return reg, T_model_from_cam

    def transform_pose(self, T: np.ndarray, grasp_pose_model: np.ndarray) -> np.ndarray:
        assert grasp_pose_model.shape == (6,)
        xyz = grasp_pose_model[:3]
        rpy = grasp_pose_model[3:6]
        def rpy_to_R(rx, ry, rz):
            sx, cx = np.sin(rx), np.cos(rx)
            sy, cy = np.sin(ry), np.cos(ry)
            sz, cz = np.sin(rz), np.cos(rz)
            Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
            Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
            Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
            return (Rz @ Ry @ Rx).astype(np.float32)
        def R_to_rpy(R):
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            singular = sy < 1e-6
            if not singular:
                rx = np.arctan2(R[2,1], R[2,2])
                ry = np.arctan2(-R[2,0], sy)
                rz = np.arctan2(R[1,0], R[0,0])
            else:
                rx = np.arctan2(-R[1,2], R[1,1])
                ry = np.arctan2(-R[2,0], sy)
                rz = 0.0
            return np.array([rx, ry, rz], dtype=np.float32)
        Rm = rpy_to_R(*rpy)
        Tm = np.eye(4, dtype=np.float32)
        Tm[:3, :3] = Rm
        Tm[:3, 3] = xyz
        Tout = (T @ Tm).astype(np.float32)
        R_out = Tout[:3, :3]
        t_out = Tout[:3, 3]
        rpy_out = R_to_rpy(R_out)
        return np.concatenate([t_out, rpy_out], axis=0)

    def cleanup(self):
        try:
            import cv2 as _cv
            _cv.destroyAllWindows()
        except Exception:
            pass
        try:
            if self.pipeline:
                self.pipeline.stop()
        except Exception:
            pass


