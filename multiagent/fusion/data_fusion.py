"""
Module for fusing data from multiple robots into a consistent state representation.
"""
import numpy as np
from datetime import datetime
import threading

class DataFusionEngine:
    """
    Engine for fusing data from multiple sources into a consistent state estimate.
    """
    
    def __init__(self, environment_state, confidence_threshold=0.5):
        """
        Initialize data fusion engine.
        
        Args:
            environment_state: EnvironmentState instance to update
            confidence_threshold: Minimum confidence for data to be used
        """
        self.environment_state = environment_state
        self.confidence_threshold = confidence_threshold
        self.robot_observations = {}  # Recent observations from each robot
        self.lock = threading.Lock()
    
    def add_observation(self, robot_id, observation):
        """
        Add an observation from a robot.
        
        Args:
            robot_id: ID of the robot providing the observation
            observation: Dictionary containing observed objects and their states
                         Format: {
                             "objects": {
                                 "obj_id": {
                                     "position": [x, y, z],
                                     "orientation": [qw, qx, qy, qz],  # optional
                                     "confidence": float,  # detection confidence
                                     "properties": {...}   # optional
                                 },
                                 ...
                             },
                             "timestamp": datetime or string
                         }
        """
        with self.lock:
            # Ensure timestamp is datetime
            if isinstance(observation.get("timestamp"), str):
                try:
                    observation["timestamp"] = datetime.fromisoformat(observation["timestamp"])
                except (ValueError, TypeError):
                    observation["timestamp"] = datetime.now()
            elif observation.get("timestamp") is None:
                observation["timestamp"] = datetime.now()
            
            # Store observation
            self.robot_observations[robot_id] = observation
            
            # Update environment state based on this observation
            self._update_environment_state()
    
    def _update_environment_state(self):
        """Update the environment state based on all recent observations."""
        # Process object observations
        all_object_observations = {}
        
        # Collect all observations for each object
        for robot_id, observation in self.robot_observations.items():
            for obj_id, obj_data in observation.get("objects", {}).items():
                if obj_data.get("confidence", 1.0) < self.confidence_threshold:
                    continue  # Skip low-confidence detections
                
                if obj_id not in all_object_observations:
                    all_object_observations[obj_id] = []
                
                all_object_observations[obj_id].append({
                    "robot_id": robot_id,
                    "data": obj_data,
                    "timestamp": observation.get("timestamp", datetime.now())
                })
        
        # Fuse observations for each object
        for obj_id, observations in all_object_observations.items():
            # Simple fusion: use most recent high-confidence observation
            # This can be replaced with more sophisticated fusion algorithms
            best_observation = max(observations, key=lambda x: x["timestamp"])
            
            position = best_observation["data"].get("position")
            orientation = best_observation["data"].get("orientation")
            properties = best_observation["data"].get("properties")
            
            if position is not None:
                self.environment_state.update_object(
                    obj_id, position, orientation, properties
                )
    
    def fuse_robot_positions(self, external_robot_positions=None):
        """
        Fuse robot position information.
        
        Args:
            external_robot_positions: Optional dictionary mapping robot_id to
                                     position data from external sources
        """
        # Collect robot positions from observations
        robot_positions = {}
        
        # Get positions from robot observations
        for robot_id, observation in self.robot_observations.items():
            if "robot_state" in observation and "position" in observation["robot_state"]:
                robot_positions[robot_id] = {
                    "position": observation["robot_state"]["position"],
                    "orientation": observation["robot_state"].get("orientation"),
                    "timestamp": observation.get("timestamp", datetime.now())
                }
        
        # Add external position data if provided
        if external_robot_positions:
            for robot_id, position_data in external_robot_positions.items():
                # Only override if external data is newer
                if (robot_id not in robot_positions or
                    position_data.get("timestamp", datetime.now()) > 
                    robot_positions[robot_id].get("timestamp", datetime.min)):
                    robot_positions[robot_id] = position_data
        
        # Update environment state with robot positions
        for robot_id, position_data in robot_positions.items():
            self.environment_state.update_robot(
                robot_id,
                position_data["position"],
                position_data.get("orientation"),
                position_data.get("properties")
            )