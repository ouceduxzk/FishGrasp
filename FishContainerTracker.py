#!/usr/bin/env python3
"""
Fish Container Tracker Module

This module tracks individual fish weight, pose, count, and determines when the container is full.
It maintains a database of fish information and provides status updates.

Features:
- Track individual fish weight and initial pose
- Count total fish added
- Calculate total weight
- Determine when container is full (12.5kg limit)
- Provide status and statistics
"""

import json
import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import threading


@dataclass
class FishRecord:
    """Data class for individual fish record"""
    fish_id: int
    timestamp: str
    weight_kg: float
    initial_pose: List[float]  # [x, y, z, rx, ry, rz]
    final_pose: Optional[List[float]] = None  # [x, y, z, rx, ry, rz]
    grasp_angle: float = 0.0  # Angle used for grasping
    status: str = "grasped"  # grasped, placed, failed
    processing_time: float = 0.0  # Time taken to process this fish


class FishContainerTracker:
    """
    Tracks fish in container with weight, pose, and count management.
    
    This class maintains a database of fish records and provides methods to:
    - Add new fish records
    - Calculate total weight and count
    - Check if container is full
    - Generate status reports
    - Save/load tracking data
    """
    
    def __init__(self, max_weight_kg: float = 12.5, data_file: str = "fish_tracking_data.json"):
        """
        Initialize the fish container tracker.
        
        Args:
            max_weight_kg: Maximum weight capacity of the container (default: 12.5kg)
            data_file: Path to JSON file for persistent data storage
        """
        self.max_weight_kg = max_weight_kg
        self.data_file = data_file
        self.fish_records: List[FishRecord] = []
        self.current_fish_id = 0
        self.lock = threading.Lock()  # Thread safety for concurrent access
        
        # Statistics
        self.total_weight_kg = 0.0
        self.total_count = 0
        self.successful_placements = 0
        self.failed_placements = 0
        
        # Load existing data if available
        self.load_data()
        
        print(f"🐟 Fish Container Tracker initialized")
        print(f"   Max capacity: {self.max_weight_kg}kg")
        print(f"   Current weight: {self.total_weight_kg:.3f}kg")
        print(f"   Current count: {self.total_count}")
        print(f"   Data file: {self.data_file}")
    
    def add_fish(self, weight_kg: float, initial_pose: List[float], 
                 grasp_angle: float = 0.0) -> int:
        """
        Add a new fish record to the tracker.
        
        Args:
            weight_kg: Weight of the fish in kilograms
            initial_pose: Initial pose [x, y, z, rx, ry, rz] in mm and radians
            grasp_angle: Angle used for grasping in radians
            
        Returns:
            fish_id: Unique identifier for this fish
        """
        with self.lock:
            self.current_fish_id += 1
            fish_id = self.current_fish_id
            
            # Create new fish record
            fish_record = FishRecord(
                fish_id=fish_id,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                weight_kg=weight_kg,
                initial_pose=initial_pose.copy(),
                grasp_angle=grasp_angle,
                status="grasped"
            )
            
            # Add to records
            self.fish_records.append(fish_record)
            
            # Update statistics
            self.total_weight_kg += weight_kg
            self.total_count += 1
            
            print(f"🐟 Added fish #{fish_id}: {weight_kg:.3f}kg, pose: {initial_pose}")
            print(f"   Total: {self.total_count} fish, {self.total_weight_kg:.3f}kg")
            
            # Save data
            self.save_data()
            
            return fish_id
    
    def update_fish_status(self, fish_id: int, status: str, 
                          final_pose: Optional[List[float]] = None,
                          processing_time: float = 0.0) -> bool:
        """
        Update the status of a fish record.
        
        Args:
            fish_id: ID of the fish to update
            status: New status ("grasped", "placed", "failed")
            final_pose: Final pose after placement (optional)
            processing_time: Time taken to process this fish
            
        Returns:
            success: True if fish was found and updated
        """
        with self.lock:
            for fish in self.fish_records:
                if fish.fish_id == fish_id:
                    fish.status = status
                    if final_pose is not None:
                        fish.final_pose = final_pose.copy()
                    fish.processing_time = processing_time
                    
                    # Update counters
                    if status == "placed":
                        self.successful_placements += 1
                    elif status == "failed":
                        self.failed_placements += 1
                    
                    print(f"🐟 Updated fish #{fish_id}: {status}")
                    self.save_data()
                    return True
            
            print(f"⚠️  Fish #{fish_id} not found for status update")
            return False
    
    def is_container_full(self) -> bool:
        """
        Check if the container has reached maximum capacity.
        
        Returns:
            is_full: True if container is full or over capacity
        """
        return self.total_weight_kg >= self.max_weight_kg
    
    def get_remaining_capacity(self) -> float:
        """
        Get remaining capacity in the container.
        
        Returns:
            remaining_kg: Remaining capacity in kilograms
        """
        return max(0.0, self.max_weight_kg - self.total_weight_kg)
    
    def get_capacity_percentage(self) -> float:
        """
        Get current capacity as a percentage.
        
        Returns:
            percentage: Capacity percentage (0-100)
        """
        return (self.total_weight_kg / self.max_weight_kg) * 100.0
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive status summary.
        
        Returns:
            status: Dictionary containing all tracking information
        """
        with self.lock:
            return {
                "total_fish": self.total_count,
                "total_weight_kg": round(self.total_weight_kg, 3),
                "max_capacity_kg": self.max_weight_kg,
                "remaining_capacity_kg": round(self.get_remaining_capacity(), 3),
                "capacity_percentage": round(self.get_capacity_percentage(), 1),
                "is_full": self.is_container_full(),
                "successful_placements": self.successful_placements,
                "failed_placements": self.failed_placements,
                "success_rate": round(self.successful_placements / max(1, self.total_count) * 100, 1),
                "average_weight_kg": round(self.total_weight_kg / max(1, self.total_count), 3),
                "current_fish_id": self.current_fish_id
            }
    
    def print_status(self):
        """Print current status to console."""
        status = self.get_status_summary()
        
        print("\n" + "="*60)
        print("🐟 FISH CONTAINER STATUS")
        print("="*60)
        print(f"Total Fish: {status['total_fish']}")
        print(f"Total Weight: {status['total_weight_kg']}kg / {status['max_capacity_kg']}kg")
        print(f"Capacity: {status['capacity_percentage']}%")
        print(f"Remaining: {status['remaining_capacity_kg']}kg")
        print(f"Container Full: {'YES' if status['is_full'] else 'NO'}")
        print(f"Success Rate: {status['success_rate']}% ({status['successful_placements']}/{status['total_fish']})")
        print(f"Average Weight: {status['average_weight_kg']}kg")
        print("="*60)
    
    def get_recent_fish(self, count: int = 5) -> List[FishRecord]:
        """
        Get the most recent fish records.
        
        Args:
            count: Number of recent records to return
            
        Returns:
            recent_fish: List of recent FishRecord objects
        """
        with self.lock:
            return self.fish_records[-count:] if self.fish_records else []
    
    def reset_container(self, confirm: bool = False) -> bool:
        """
        Reset the container (clear all data).
        
        Args:
            confirm: Must be True to actually reset
            
        Returns:
            success: True if reset was successful
        """
        if not confirm:
            print("⚠️  Reset requires confirmation. Call with confirm=True")
            return False
        
        with self.lock:
            self.fish_records.clear()
            self.current_fish_id = 0
            self.total_weight_kg = 0.0
            self.total_count = 0
            self.successful_placements = 0
            self.failed_placements = 0
            
            print("🔄 Container reset successfully")
            self.save_data()
            return True
    
    def save_data(self):
        """Save tracking data to JSON file."""
        try:
            with self.lock:
                data = {
                    "max_weight_kg": self.max_weight_kg,
                    "current_fish_id": self.current_fish_id,
                    "total_weight_kg": self.total_weight_kg,
                    "total_count": self.total_count,
                    "successful_placements": self.successful_placements,
                    "failed_placements": self.failed_placements,
                    "fish_records": [asdict(fish) for fish in self.fish_records],
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(self.data_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
        except Exception as e:
            print(f"⚠️  Failed to save tracking data: {e}")
    
    def load_data(self):
        """Load tracking data from JSON file."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load basic info
                self.max_weight_kg = data.get("max_weight_kg", 12.5)
                self.current_fish_id = data.get("current_fish_id", 0)
                self.total_weight_kg = data.get("total_weight_kg", 0.0)
                self.total_count = data.get("total_count", 0)
                self.successful_placements = data.get("successful_placements", 0)
                self.failed_placements = data.get("failed_placements", 0)
                
                # Load fish records
                self.fish_records = []
                for fish_data in data.get("fish_records", []):
                    fish_record = FishRecord(**fish_data)
                    self.fish_records.append(fish_record)
                
                print(f"📁 Loaded tracking data: {len(self.fish_records)} fish records")
                
        except Exception as e:
            print(f"⚠️  Failed to load tracking data: {e}")
            print("   Starting with empty container")
    
    def export_data(self, filename: Optional[str] = None) -> str:
        """
        Export tracking data to a CSV file.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            filepath: Path to the exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fish_tracking_export_{timestamp}.csv"
        
        try:
            import csv
            
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'fish_id', 'timestamp', 'weight_kg', 'status',
                    'initial_x', 'initial_y', 'initial_z', 'initial_rx', 'initial_ry', 'initial_rz',
                    'final_x', 'final_y', 'final_z', 'final_rx', 'final_ry', 'final_rz',
                    'grasp_angle', 'processing_time'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                with self.lock:
                    for fish in self.fish_records:
                        row = {
                            'fish_id': fish.fish_id,
                            'timestamp': fish.timestamp,
                            'weight_kg': fish.weight_kg,
                            'status': fish.status,
                            'initial_x': fish.initial_pose[0] if len(fish.initial_pose) > 0 else '',
                            'initial_y': fish.initial_pose[1] if len(fish.initial_pose) > 1 else '',
                            'initial_z': fish.initial_pose[2] if len(fish.initial_pose) > 2 else '',
                            'initial_rx': fish.initial_pose[3] if len(fish.initial_pose) > 3 else '',
                            'initial_ry': fish.initial_pose[4] if len(fish.initial_pose) > 4 else '',
                            'initial_rz': fish.initial_pose[5] if len(fish.initial_pose) > 5 else '',
                            'final_x': fish.final_pose[0] if fish.final_pose and len(fish.final_pose) > 0 else '',
                            'final_y': fish.final_pose[1] if fish.final_pose and len(fish.final_pose) > 1 else '',
                            'final_z': fish.final_pose[2] if fish.final_pose and len(fish.final_pose) > 2 else '',
                            'final_rx': fish.final_pose[3] if fish.final_pose and len(fish.final_pose) > 3 else '',
                            'final_ry': fish.final_pose[4] if fish.final_pose and len(fish.final_pose) > 4 else '',
                            'final_rz': fish.final_pose[5] if fish.final_pose and len(fish.final_pose) > 5 else '',
                            'grasp_angle': fish.grasp_angle,
                            'processing_time': fish.processing_time
                        }
                        writer.writerow(row)
            
            print(f"📊 Exported tracking data to: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️  Failed to export data: {e}")
            return ""


def main():
    """Test the FishContainerTracker module."""
    print("Testing FishContainerTracker...")
    
    # Create tracker
    tracker = FishContainerTracker(max_weight_kg=12.5)
    
    # Add some test fish
    test_fish = [
        (0.5, [100, 200, 300, 0, 0, 0], 0.1),
        (0.8, [150, 250, 350, 0, 0, 0.5], 0.2),
        (1.2, [200, 300, 400, 0, 0, 1.0], 0.3),
    ]
    
    for weight, pose, angle in test_fish:
        fish_id = tracker.add_fish(weight, pose, angle)
        tracker.update_fish_status(fish_id, "placed", pose, 2.5)
    
    # Print status
    tracker.print_status()
    
    # Test container full check
    print(f"Container full: {tracker.is_container_full()}")
    print(f"Remaining capacity: {tracker.get_remaining_capacity()}kg")
    
    # Export data
    tracker.export_data("test_fish_tracking.csv")


if __name__ == "__main__":
    main()
