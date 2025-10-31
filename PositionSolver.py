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
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math


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

    @staticmethod
    def load_grid_params(json_path: str):
        """
        Convenience loader for grid/box planning parameters used by PositionSolver.
        Returns a dict with keys: rows, cols, order, box_size_mm, corner_xy_mm,
        x_margin_mm, y_margin_mm, approach_factor, place_x_mm, place_factor.
        """
        import json
        from pathlib import Path
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        grid = data.get("grid", {})
        box = data.get("box", {})
        rule = data.get("waypoint_rule", {})

        size = box.get("size_mm", [200.0, 300.0])
        corner = box.get("top_right_corner_mm", [0.0, 0.0])

        return {
            "rows": int(grid.get("rows", 1)),
            "cols": int(grid.get("cols", 1)),
            "order": str(grid.get("order", "row-major")),
            "box_size_mm": (float(size[0]), float(size[1])),
            "corner_xy_mm": (float(corner[0]), float(corner[1])),
            "x_margin_mm": float(box.get("x_margin_mm", 0.0)),
            "y_margin_mm": float(box.get("y_margin_mm", 0.0)),
            "approach_factor": float(rule.get("approach_factor", 0.8)),
            "place_x_mm": float(rule.get("place_x_mm", 0.0)),
            "place_factor": float(rule.get("place_factor", 0.1)),
        }


@dataclass
class FishPlacement:
    """Fish placement information"""
    x_mm: float
    y_mm: float
    z_mm: float
    layer: int
    grid_x: int
    grid_y: int
    fish_id: int
    timestamp: str


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
        

    def plan_linear_row_centers(
        self,
        num_fish: int,
        box_size_mm: Tuple[float, float] = (600.0, 4000.0),
        corner_xy_mm: Tuple[float, float] = (-300.0, -320.0),
        x_margin_mm: float = 0.0,
        y_center_bias_mm: float = 0.0,
    ) -> List[Tuple[float, float]]:
        """
        Compute centers for placing fish in a single column along x, where fish
        body length spans the y direction.

        Coordinate convention (matches user's sketch):
        - Robot base at (0, 0)
        - x increases downward
        - y increases to the right
        - The provided corner is the box's top-right corner

        Args:
            num_fish: Number of fish to place.
            box_size_mm: (height_x_mm, width_y_mm) of the box.
            corner_xy_mm: (x, y) of the top-right corner of the box relative to robot base.
            x_margin_mm: Optional margin from top/bottom edges along x.
            y_center_bias_mm: Optional offset to move the centers along y.

        Returns:
            List of (x_mm, y_mm) centers for each fish, length == num_fish.
        """
        if num_fish <= 0:
            return []

        height_x_mm, width_y_mm = box_size_mm
        corner_x, corner_y = corner_xy_mm

        # Effective x-range inside the box after applying top/bottom margins
        x_min = corner_x + x_margin_mm  # top edge (moving downward is +x)
        x_max = corner_x + height_x_mm - x_margin_mm
        if x_max <= x_min:
            # Degenerate case: no usable space
            center_x = (corner_x + corner_x + height_x_mm) / 2.0
            center_y = corner_y - width_y_mm / 2.0 + y_center_bias_mm
            return [(center_x, center_y) for _ in range(num_fish)]

        # Even spacing along x inside [x_min, x_max]
        pitch_x = (x_max - x_min) / float(num_fish)
        centers_x = [x_min + (i + 0.5) * pitch_x for i in range(num_fish)]

        # Fish body spans the y direction; place all at the center of the box in y
        # The box extends to the LEFT from the top-right corner, so subtract width
        center_y = corner_y - width_y_mm / 2.0 + y_center_bias_mm

        return [(cx, center_y) for cx in centers_x]

    def plan_grid_centers(
        self,
        rows: int,
        cols: int,
        box_size_mm: Tuple[float, float] = (600.0, 400.0),
        corner_xy_mm: Tuple[float, float] = (-300.0, -320.0),
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
        box_size_mm: Tuple[float, float] = (600.0, 4000.0),
        corner_xy_mm: Tuple[float, float] = (-300.0, -320.0),
        x_margin_mm: float = 0.0,
        y_margin_mm: float = 0.0,
        approach_factor: float = 0.8,
        place_x_mm: float = 0.0,
        place_factor: float = 0.1,
        order: str = "row-major",
    ) -> str:
        """
        Generate an r×c grid of centers and save JSON with two waypoints per id:
        {"1": [[x, 0.8*y], [0, 0.1*y]], ...}
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
            out[str(i)] = [
                [float(cx), float(approach_factor * cy)],
                [float(place_x_mm), float(place_factor * cy)],
            ]
        from pathlib import Path
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)
    
    def print_placement_status(self):
        """Print current placement status."""
        stats = self.get_placement_statistics()
        
        print("\n" + "="*50)
        print("📍 PLACEMENT STATUS")
        print("="*50)
        print(f"Total Fish Placed: {stats['total_fish']}")
        print(f"Layers Used: {stats['layers_used']}")
        print(f"Current Layer: {stats['current_layer']}")
        print(f"Grid Utilization: {stats['grid_utilization']}%")
        print(f"Average Layer: {stats['average_layer']}")
        print("="*50)
    
    def get_recent_placements(self, count: int = 5) -> List[FishPlacement]:
        """
        Get recent placement records.
        
        Args:
            count: Number of recent records to return
            
        Returns:
            recent_placements: List of recent FishPlacement objects
        """
        return self.placement_history[-count:] if self.placement_history else []
    
    def reset_placements(self, confirm: bool = False) -> bool:
        """
        Reset all placements (clear the grid).
        
        Args:
            confirm: Must be True to actually reset
            
        Returns:
            success: True if reset was successful
        """
        if not confirm:
            print("⚠️  Reset requires confirmation. Call with confirm=True")
            return False
        
        self.placement_grid.fill(False)
        self.placement_history.clear()
        self.current_layer = 0
        
        print("🔄 Placements reset successfully")
        return True
    
    def get_container_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get container bounds (min_x, max_x, min_y, max_y, min_z, max_z).
        
        Returns:
            bounds: Tuple of (min_x, max_x, min_y, max_y, min_z, max_z) in mm
        """
        return (
            0.0, self.config.width_mm,
            0.0, self.config.height_mm,
            self.config.base_height_mm, self.config.base_height_mm + self.config.depth_mm
        )
    
    def visualize_placements(self, layer: Optional[int] = None) -> str:
        """
        Create a text visualization of current placements.
        
        Args:
            layer: Specific layer to visualize (None for current layer)
            
        Returns:
            visualization: String representation of the grid
        """
        if layer is None:
            layer = self.current_layer
        
        if layer >= self.placement_grid.shape[0]:
            return f"Layer {layer} does not exist"
        
        grid = self.placement_grid[layer]
        rows, cols = grid.shape
        
        result = f"\nLayer {layer} ({rows}x{cols} grid):\n"
        result += "  " + "".join(f"{i%10}" for i in range(cols)) + "\n"
        
        for y in range(rows):
            result += f"{y%10} " + "".join("█" if grid[y, x] else "·" for x in range(cols)) + "\n"
        
        return result


def main():
    """Test the PositionSolver module."""
    print("Testing PositionSolver...")
    

    # Example A: ContainerConfig from JSON
    try:
        cfg = ContainerConfig.from_params_json("configs/fish_grid_params.json")
        print(f"Loaded ContainerConfig from JSON: width={cfg.width_mm}, height={cfg.height_mm}, depth={cfg.depth_mm}")
    except Exception as e:
        print(f"Failed to load ContainerConfig from JSON, using defaults: {e}")
        cfg = ContainerConfig()

    # Example A: compute linear row centers
    n = 6
    centers = PositionSolver(cfg).plan_linear_row_centers(
        num_fish=n,
        box_size_mm=(cfg.height_mm, cfg.width_mm),
        corner_xy_mm=(40.0, -300.0),
        x_margin_mm=0.0,
        y_center_bias_mm=0.0,
    )
    print(f"\nPlanned {n} centers (x, y) mm (linear):")
    for i, (cx, cy) in enumerate(centers):
        print(f"  {i+1}: ({cx:.1f}, {cy:.1f})")

    # # Example B: compute grid centers r×c
    # r, c = 3, 2
    # grid_centers = PositionSolver(cfg).plan_grid_centers(
    #     rows=r,
    #     cols=c,
    #     box_size_mm=(cfg.height_mm, cfg.width_mm),
    #     corner_xy_mm=(40.0, -300.0),
    #     x_margin_mm=0.0,
    #     y_margin_mm=0.0,
    #     order="row-major",
    # )
    # print(f"\nPlanned grid centers ({r}x{c}) (x, y) mm:")
    # for i, (cx, cy) in enumerate(grid_centers):
    #     print(f"  {i+1}: ({cx:.1f}, {cy:.1f})")

    # # Example C: export grid waypoints JSON for realtime loader
    # # Generates keys 1..(r*c) with [[x, 0.8*y], [0, 0.1*y]]
    # try:
    #     path = PositionSolver(cfg).export_grid_waypoints_json(
    #         file_path="configs/fish_paths.json",
    #         rows=r,
    #         cols=c,
    #         box_size_mm=(cfg.height_mm, cfg.width_mm),
    #         corner_xy_mm=(40.0, -300.0),
    #         x_margin_mm=0.0,
    #         y_margin_mm=0.0,
    #         approach_factor=0.8,
    #         place_x_mm=0.0,
    #         place_factor=0.1,
    #         order="row-major",
    #     )
    #     print(f"\nExported grid waypoints JSON to: {path}")
    # except Exception as e:
    #     print(f"Failed exporting grid JSON: {e}")


if __name__ == "__main__":
    main()
