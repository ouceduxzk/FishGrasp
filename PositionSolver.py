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
    

    # Example: compute linear row centers inside a 600mm x 4000mm box whose
    # top-right corner is at (-300mm, -320mm). Place N fish along x.
    n = 6
    centers = PositionSolver(ContainerConfig()).plan_linear_row_centers(
        num_fish=n,
        box_size_mm=(600.0, 400.0),
        corner_xy_mm=(40.0, -300.0),
        x_margin_mm=0.0,
        y_center_bias_mm=0.0,
    )
    print(f"\nPlanned {n} centers (x, y) mm:")
    for i, (cx, cy) in enumerate(centers):
        print(f"  {i+1}: ({cx:.1f}, {cy:.1f})")


if __name__ == "__main__":
    main()
