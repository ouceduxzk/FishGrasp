#!/usr/bin/env python3
"""
Position Solver Module

This module predicts the final XYZ position where a fish will be placed in the container.
It uses a grid-based placement strategy to optimize space utilization.

Features:
- Grid-based placement strategy
- Collision avoidance
- Space optimization
- Configurable container dimensions
- Placement pattern management
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


@dataclass
class ContainerConfig:
    """Container configuration parameters"""
    width_mm: float = 300.0      # Container width in mm
    height_mm: float = 200.0     # Container height in mm
    depth_mm: float = 150.0      # Container depth in mm
    grid_spacing_mm: float = 30.0  # Grid spacing in mm
    margin_mm: float = 20.0      # Margin from container edges
    base_height_mm: float = 0.0  # Base height of container

    @classmethod
    def from_params_json(
        cls,
        json_path: str,
        default_depth_mm: float = 150.0,
        default_grid_spacing_mm: float = 30.0,
        default_margin_mm: float = 20.0,
        default_base_height_mm: float = 0.0,
    ) -> "ContainerConfig":
        """
        Load container-related parameters from fish_grid_params.json.

        Mapping rules:
        - box.size_mm is [height_x_mm, width_y_mm]; map to height_mm and width_mm
        - depth/grid_spacing/margin/base_height may be absent; fallback to defaults
        """
        import json
        from pathlib import Path
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))

        box = data.get("box", {})
        size = box.get("size_mm", [200.0, 300.0])
        # size_mm: [height_x_mm, width_y_mm]
        height_x_mm = float(size[0]) if len(size) > 0 else 200.0
        width_y_mm = float(size[1]) if len(size) > 1 else 300.0

        depth_mm = float(box.get("depth_mm", default_depth_mm))
        grid_spacing_mm = float(box.get("grid_spacing_mm", default_grid_spacing_mm))
        margin_mm = float(box.get("x_margin_mm", box.get("margin_mm", default_margin_mm)))
        base_height_mm = float(box.get("base_height_mm", default_base_height_mm))

        return cls(
            width_mm=width_y_mm,
            height_mm=height_x_mm,
            depth_mm=depth_mm,
            grid_spacing_mm=grid_spacing_mm,
            margin_mm=margin_mm,
            base_height_mm=base_height_mm,
        )

class PositionSolver:
    """
    Solves optimal placement positions for fish in a container.
    
    This class manages a grid-based placement system that:
    - Places fish in layers to maximize space utilization
    - Avoids collisions between fish
    - Maintains consistent spacing
    - Tracks placement history
    """
    
    def __init__(self, container_config: Optional[ContainerConfig] = None):
        """
        Initialize the position solver.
        
        Args:
            container_config: Container configuration parameters
        """
        self.config = container_config or ContainerConfig()
        
        # Calculate grid dimensions
        self.grid_width = int((self.config.width_mm - 2 * self.config.margin_mm) / self.config.grid_spacing_mm)
        self.grid_height = int((self.config.height_mm - 2 * self.config.margin_mm) / self.config.grid_spacing_mm)
        

    def plan_grid_centers(
        self,
        rows: int,
        cols: int,
        box_size_mm: Tuple[float, float] = (600.0, 400.0),
        corner_xy_mm: Tuple[float, float] = (40.0, -320.0),
        x_margin_mm: float = 0.0,
        y_margin_mm: float = 0.0,
        order: str = "row-major",
    ) -> List[Tuple[float, float]]:
        """
        Compute an r×c grid of centers inside the rectangular box.

        - rows are along x (top to bottom), cols along y (right to left).
        - The provided corner is the top-right corner of the box.
        - x increases downward; y increases to the right.

        Args:
            rows: Number of rows (along x direction).
            cols: Number of columns (along y direction).
            box_size_mm: (height_x_mm, width_y_mm) of the box.
            corner_xy_mm: (x, y) of the top-right corner.
            x_margin_mm: Margin on top/bottom sides.
            y_margin_mm: Margin on right/left sides.
            order: 'row-major' or 'col-major' for output ordering.

        Returns:
            List of (x_mm, y_mm) centers with length rows*cols.
        """
        if rows <= 0 or cols <= 0:
            return []

        height_x_mm, width_y_mm = box_size_mm
        corner_x, corner_y = corner_xy_mm

        # Effective bounds
        x_min = corner_x + x_margin_mm
        x_max = corner_x + height_x_mm - x_margin_mm
        y_right = corner_y - y_margin_mm           # right edge
        y_left = corner_y - width_y_mm + y_margin_mm  # left edge (more negative)

        if x_max <= x_min or y_right <= y_left:
            # Fallback to single center repeated
            cx = (corner_x + corner_x + height_x_mm) / 2.0
            cy = corner_y - width_y_mm / 2.0
            return [(cx, cy) for _ in range(rows * cols)]

        pitch_x = (x_max - x_min) / float(rows)
        pitch_y = (y_right - y_left) / float(cols)

        centers: List[Tuple[float, float]] = []
        if order == "row-major":
            for r in range(rows):
                cx = x_min + (r + 0.5) * pitch_x
                for c in range(cols):
                    # columns go from right to left within the box
                    cy = y_left + (c + 0.5) * pitch_y
                    centers.append((cx, cy))
        else:  # col-major
            for c in range(cols):
                cy = y_left + (c + 0.5) * pitch_y
                for r in range(rows):
                    cx = x_min + (r + 0.5) * pitch_x
                    centers.append((cx, cy))

        return centers

    def export_grid_waypoints_json(
        self,
        file_path: str,
        rows: int,
        cols: int,
        box_size_mm: Tuple[float, float] = (600.0, 400.0),
        corner_xy_mm: Tuple[float, float] = (40.0, -320.0),
        x_margin_mm: float = 0.0,
        y_margin_mm: float = 0.0,
        approach_factor: float = 0.8,
        place_x_mm: float = 0.0,
        order: str = "row-major",
    ) -> str:
        """
        Generate an r×c grid of centers and save JSON with two waypoints per id:
        {"1": [[cx, approach_factor*cy], [place_x, (1-approach_factor)*cy]], ...}
        point1 cy = approach_factor * cy_center (e.g., -520 * 0.8 = -416)
        point2 cy = (1 - approach_factor) * cy_center (e.g., -520 * 0.2 = -104)
        Returns the file path.
        """
        import json
        centers = self.plan_grid_centers(
            rows=rows,
            cols=cols,
            box_size_mm=box_size_mm,
            corner_xy_mm=corner_xy_mm,
            x_margin_mm=x_margin_mm,
            y_margin_mm=y_margin_mm,
            order=order,
        )
        out = {}
        for i, (cx, cy) in enumerate(centers, start=1):
            # point1: approach position with cy = approach_factor * cy_center
            # point2: place position with cy = (1 - approach_factor) * cy_center
            out[str(i)] = [
                [float(cx), float(approach_factor * cy)],  # point1: approach
                [float(place_x_mm), float((1.0 - approach_factor) * cy)],  # point2: place
            ]
        from pathlib import Path
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)

    def visualize_grid_waypoints(
        self,
        output_image_path: str,
        rows: int,
        cols: int,
        box_size_mm: Tuple[float, float] = (600.0, 400.0),
        corner_xy_mm: Tuple[float, float] = (40.0, -320.0),
        x_margin_mm: float = 0.0,
        y_margin_mm: float = 0.0,
        approach_factor: float = 0.8,
        place_x_mm: float = 0.0,
        order: str = "row-major",
        image_size: Tuple[int, int] = (640, 480),
    ) -> str:
        """
        Visualize grid waypoints: draw box rectangle and mark each target with its index number.
        Saves to output_image_path (e.g., configs/fish_paths.jpg) at specified image_size.
        """
        if not HAS_OPENCV:
            print("OpenCV not available, skipping visualization")
            return None

        centers = self.plan_grid_centers(
            rows=rows,
            cols=cols,
            box_size_mm=box_size_mm,
            corner_xy_mm=corner_xy_mm,
            x_margin_mm=x_margin_mm,
            y_margin_mm=y_margin_mm,
            order=order,
        )

        height_x_mm, width_y_mm = box_size_mm
        corner_x, corner_y = corner_xy_mm
        box_x_min = corner_x
        box_x_max = corner_x + height_x_mm
        box_y_min = corner_y - width_y_mm
        box_y_max = corner_y

        # Compute bounds with padding
        all_x = [cx for cx, _ in centers] + [box_x_min, box_x_max]
        all_y = [cy for _, cy in centers] + [box_y_min, box_y_max]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        padding = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 50.0)
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding

        # Create image
        img_w, img_h = image_size
        img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

        def world_to_img(x_world, y_world):
            x_img = int(((x_world - x_min) / (x_max - x_min)) * img_w)
            y_img = int(((y_world - y_min) / (y_max - y_min)) * img_h)
            return x_img, y_img

        # Draw box rectangle
        p1 = world_to_img(box_x_min, box_y_min)
        p2 = world_to_img(box_x_max, box_y_max)
        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)

        # Label corner point
        corner_img_x, corner_img_y = world_to_img(corner_x, corner_y)
        cv2.circle(img, (corner_img_x, corner_img_y), 5, (255, 0, 0), -1)
        cv2.putText(img, f"Corner: ({corner_x:.1f}, {corner_y:.1f})", 
                   (corner_img_x + 10, corner_img_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Add title and box size
        cv2.putText(img, f"Grid: {rows}x{cols} = {rows*cols} targets", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f"Box size: {height_x_mm:.1f}mm x {width_y_mm:.1f}mm", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw each target center with index and coordinates
        for idx, (cx, cy) in enumerate(centers, start=1):
            x_img, y_img = world_to_img(cx, cy)
            cv2.circle(img, (x_img, y_img), 8, (0, 255, 0), -1)
            cv2.circle(img, (x_img, y_img), 8, (0, 0, 0), 2)
            # Label with index
            cv2.putText(img, f"#{idx}", (x_img + 12, y_img - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # Label with coordinates
            cv2.putText(img, f"({cx:.1f},{cy:.1f})", (x_img + 12, y_img + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        from pathlib import Path
        p = Path(output_image_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
        return str(p)
    
    def print_placement_status(self):
        """Print current placement status."""
        print("\n" + "="*50)
        print("📍 PLACEMENT STATUS")
        print("="*50)
        print("Placement tracking is not implemented")
        print("="*50)
    
    def visualize_placements(self, layer: Optional[int] = None) -> str:
        """
        Create a text visualization of current placements.
        
        Args:
            layer: Specific layer to visualize (None for current layer)
            
        Returns:
            visualization: String representation of the grid
        """
        return "Placement visualization is not implemented"


def main():
    """Test the PositionSolver module."""
    print("Testing PositionSolver...")

    # Export waypoints JSON to specified output path if present
    try:
        import json
        from pathlib import Path
        params_path = Path("configs/fish_grid_params.json")
        if params_path.exists():
            data = json.loads(params_path.read_text(encoding="utf-8"))
            out_cfg = data.get("output", {})
            waypoints_path = out_cfg.get("waypoints_json_path", "configs/fish_paths.json")
            grid = data.get("grid", {})
            box = data.get("box", {})
            rule = data.get("waypoint_rule", {})
            
            size = box.get("size_mm", [200.0, 300.0])
            corner = box.get("top_right_corner_mm", [0.0, 0.0])
            
            grid_params = {
                "rows": int(grid.get("rows", 1)),
                "cols": int(grid.get("cols", 1)),
                "order": str(grid.get("order", "row-major")),
                "box_size_mm": (float(size[0]), float(size[1])),
                "corner_xy_mm": (float(corner[0]), float(corner[1])),
                "x_margin_mm": float(box.get("x_margin_mm", 0.0)),
                "y_margin_mm": float(box.get("y_margin_mm", 0.0)),
                "approach_factor": float(rule.get("approach_factor", 0.8)),
                "place_x_mm": float(rule.get("place_x_mm", 0.0)),
            }
        else:
            waypoints_path = "configs/fish_paths.json"
            grid_params = {
                "rows": 1,
                "cols": 1,
                "order": "row-major",
                "box_size_mm": (200.0, 300.0),
                "corner_xy_mm": (40.0, -320.0),
                "x_margin_mm": 0.0,
                "y_margin_mm": 0.0,
                "approach_factor": 0.8,
                "place_x_mm": 0.0,
            }
        
        try:
            cfg = ContainerConfig.from_params_json(str(params_path))
        except Exception:
            cfg = ContainerConfig()
        
        solver = PositionSolver(cfg)
        # Extract all parameters once to ensure consistency
        rows = int(grid_params.get("rows", 1))
        cols = int(grid_params.get("cols", 1))
        box_size = grid_params.get("box_size_mm", (cfg.height_mm, cfg.width_mm))
        corner_xy = grid_params.get("corner_xy_mm", (40.0, -300.0))
        x_margin = float(grid_params.get("x_margin_mm", 0.0))
        y_margin = float(grid_params.get("y_margin_mm", 0.0))
        approach_factor_val = float(grid_params.get("approach_factor", 0.8))
        place_x = float(grid_params.get("place_x_mm", 0.0))
        order_val = str(grid_params.get("order", "row-major"))
        
        # Calculate centers using the SAME method as export (for verification)
        centers = solver.plan_grid_centers(
            rows=rows,
            cols=cols,
            box_size_mm=box_size,
            corner_xy_mm=corner_xy,
            x_margin_mm=x_margin,
            y_margin_mm=y_margin,
            order=order_val,
        )
        print(f"\nPlanned {len(centers)} centers (x, y) mm (grid: {rows}x{cols}):")
        for i, (cx, cy) in enumerate(centers, start=1):
            print(f"  {i}: ({cx:.1f}, {cy:.1f})")
        
        # Export waypoints JSON using the same parameters
        solver.export_grid_waypoints_json(
            file_path=waypoints_path,
            rows=rows,
            cols=cols,
            box_size_mm=box_size,
            corner_xy_mm=corner_xy,
            x_margin_mm=x_margin,
            y_margin_mm=y_margin,
            approach_factor=approach_factor_val,
            place_x_mm=place_x,
            order=order_val,
        )
        print(f"Waypoints exported to: {waypoints_path}")
        
        # Generate visualization image using the EXACT same parameters
        try:
            vis_path = "configs/latest_fish_paths.jpg"
            solver.visualize_grid_waypoints(
                output_image_path=vis_path,
                rows=rows,
                cols=cols,
                box_size_mm=box_size,
                corner_xy_mm=corner_xy,
                x_margin_mm=x_margin,
                y_margin_mm=y_margin,
                approach_factor=approach_factor_val,
                place_x_mm=place_x,
                order=order_val,
                image_size=(640, 480),
            )
            print(f"Visualization saved to: {vis_path}")
        except Exception as e:
            print(f"Failed to generate visualization: {e}")
    except Exception as e:
        print(f"Failed to export waypoints JSON: {e}")



if __name__ == "__main__":
    main()
