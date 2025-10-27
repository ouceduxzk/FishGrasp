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
        
        # Placement grid (3D: layer, y, x)
        self.placement_grid = np.zeros((10, self.grid_height, self.grid_width), dtype=bool)
        
        # Placement history
        self.placement_history: List[FishPlacement] = []
        
        # Current layer being filled
        self.current_layer = 0
        
        print(f"🏗️  Position Solver initialized")
        print(f"   Container: {self.config.width_mm}x{self.config.height_mm}x{self.config.depth_mm}mm")
        print(f"   Grid: {self.grid_width}x{self.grid_height} cells")
        print(f"   Spacing: {self.config.grid_spacing_mm}mm")
    
    def find_optimal_position(self, fish_id: int, fish_size_mm: Tuple[float, float, float] = (50, 30, 20)) -> Optional[FishPlacement]:
        """
        Find the optimal position for placing a fish.
        
        Args:
            fish_id: Unique identifier for the fish
            fish_size_mm: Fish dimensions (width, height, depth) in mm
            
        Returns:
            placement: FishPlacement object with position information, or None if no space available
        """
        # Calculate required grid cells (with some margin)
        cells_needed_x = max(1, int(fish_size_mm[0] / self.config.grid_spacing_mm) + 1)
        cells_needed_y = max(1, int(fish_size_mm[1] / self.config.grid_spacing_mm) + 1)
        
        # Try to find space in current layer first
        for layer in range(self.current_layer, min(self.current_layer + 3, self.placement_grid.shape[0])):
            for y in range(self.grid_height - cells_needed_y + 1):
                for x in range(self.grid_width - cells_needed_x + 1):
                    if self._is_space_available(layer, x, y, cells_needed_x, cells_needed_y):
                        # Found space, mark it as occupied
                        self._mark_space_occupied(layer, x, y, cells_needed_x, cells_needed_y)
                        
                        # Calculate actual position in mm
                        actual_x = self.config.margin_mm + x * self.config.grid_spacing_mm + self.config.grid_spacing_mm / 2
                        actual_y = self.config.margin_mm + y * self.config.grid_spacing_mm + self.config.grid_spacing_mm / 2
                        actual_z = self.config.base_height_mm + layer * 25.0  # 25mm layer height
                        
                        # Create placement record
                        placement = FishPlacement(
                            x_mm=actual_x,
                            y_mm=actual_y,
                            z_mm=actual_z,
                            layer=layer,
                            grid_x=x,
                            grid_y=y,
                            fish_id=fish_id,
                            timestamp=self._get_timestamp()
                        )
                        
                        # Add to history
                        self.placement_history.append(placement)
                        
                        # Update current layer if needed
                        if layer > self.current_layer:
                            self.current_layer = layer
                        
                        print(f"📍 Placed fish #{fish_id} at ({actual_x:.1f}, {actual_y:.1f}, {actual_z:.1f})mm, layer {layer}")
                        return placement
        
        print(f"⚠️  No space available for fish #{fish_id}")
        return None
    
    def _is_space_available(self, layer: int, start_x: int, start_y: int, 
                          cells_x: int, cells_y: int) -> bool:
        """Check if space is available in the grid."""
        if (layer >= self.placement_grid.shape[0] or 
            start_x + cells_x > self.grid_width or 
            start_y + cells_y > self.grid_height):
            return False
        
        # Check if any cells in the area are occupied
        return not np.any(self.placement_grid[layer, start_y:start_y + cells_y, start_x:start_x + cells_x])
    
    def _mark_space_occupied(self, layer: int, start_x: int, start_y: int, 
                           cells_x: int, cells_y: int):
        """Mark space as occupied in the grid."""
        self.placement_grid[layer, start_y:start_y + cells_y, start_x:start_x + cells_x] = True
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def get_placement_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about current placements.
        
        Returns:
            stats: Dictionary containing placement statistics
        """
        if not self.placement_history:
            return {
                "total_fish": 0,
                "layers_used": 0,
                "grid_utilization": 0.0,
                "average_layer": 0.0
            }
        
        total_fish = len(self.placement_history)
        layers_used = max(p.layer for p in self.placement_history) + 1
        average_layer = sum(p.layer for p in self.placement_history) / total_fish
        
        # Calculate grid utilization
        occupied_cells = np.sum(self.placement_grid)
        total_cells = self.placement_grid.size
        grid_utilization = (occupied_cells / total_cells) * 100.0
        
        return {
            "total_fish": total_fish,
            "layers_used": layers_used,
            "grid_utilization": round(grid_utilization, 1),
            "average_layer": round(average_layer, 1),
            "current_layer": self.current_layer
        }
    
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
    
    # Create solver with custom config
    config = ContainerConfig(
        width_mm=300.0,
        height_mm=200.0,
        depth_mm=150.0,
        grid_spacing_mm=30.0,
        margin_mm=20.0
    )
    
    solver = PositionSolver(config)
    
    # Test placing some fish
    test_fish = [
        (1, (50, 30, 20)),
        (2, (40, 25, 15)),
        (3, (60, 35, 25)),
        (4, (45, 28, 18)),
    ]
    
    for fish_id, size in test_fish:
        placement = solver.find_optimal_position(fish_id, size)
        if placement:
            print(f"Fish #{fish_id}: ({placement.x_mm:.1f}, {placement.y_mm:.1f}, {placement.z_mm:.1f})mm")
        else:
            print(f"Fish #{fish_id}: No space available")
    
    # Print statistics
    solver.print_placement_status()
    
    # Show visualization
    print(solver.visualize_placements(0))


if __name__ == "__main__":
    main()
