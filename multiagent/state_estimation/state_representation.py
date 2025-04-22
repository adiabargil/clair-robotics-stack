"""
Module for representing the shared state in a multi-agent system.
"""
import numpy as np
from datetime import datetime
import threading

class EnvironmentState:
    """
    Class representing the shared environment state.
    """
    
    def __init__(self):
        """Initialize an empty environment state."""
        self.objects = {}  # Dictionary mapping object_id to object state
        self.robots = {}   # Dictionary mapping robot_id to robot state
        self.timestamp = datetime.now()
        self.lock = threading.Lock()
    
    def update_object(self, object_id, position, orientation=None, properties=None):
        """
        Update an object's state.
        
        Args:
            object_id: Unique identifier for the object
            position: 3D position (x, y, z)
            orientation: Optional orientation (e.g., quaternion)
            properties: Optional dictionary of additional properties
        """
        with self.lock:
            if object_id not in self.objects:
                self.objects[object_id] = {}
            
            self.objects[object_id]["position"] = np.array(position)
            
            if orientation is not None:
                self.objects[object_id]["orientation"] = np.array(orientation)
            
            if properties is not None:
                if "properties" not in self.objects[object_id]:
                    self.objects[object_id]["properties"] = {}
                self.objects[object_id]["properties"].update(properties)
            
            self.objects[object_id]["last_updated"] = datetime.now()
            self.timestamp = datetime.now()
    
    def update_robot(self, robot_id, position, orientation=None, properties=None):
        """
        Update a robot's state.
        
        Args:
            robot_id: Unique identifier for the robot
            position: 3D position (x, y, z)
            orientation: Optional orientation (e.g., quaternion)
            properties: Optional dictionary of additional properties
        """
        with self.lock:
            if robot_id not in self.robots:
                self.robots[robot_id] = {}
            
            self.robots[robot_id]["position"] = np.array(position)
            
            if orientation is not None:
                self.robots[robot_id]["orientation"] = np.array(orientation)
            
            if properties is not None:
                if "properties" not in self.robots[robot_id]:
                    self.robots[robot_id]["properties"] = {}
                self.robots[robot_id]["properties"].update(properties)
            
            self.robots[robot_id]["last_updated"] = datetime.now()
            self.timestamp = datetime.now()
    
    def get_object(self, object_id):
        """Get an object's state."""
        with self.lock:
            return self.objects.get(object_id, None)
    
    def get_robot(self, robot_id):
        """Get a robot's state."""
        with self.lock:
            return self.robots.get(robot_id, None)
    
    def get_all_objects(self):
        """Get states of all objects."""
        with self.lock:
            return self.objects.copy()
    
    def get_all_robots(self):
        """Get states of all robots."""
        with self.lock:
            return self.robots.copy()
    
    def serialize(self):
        """
        Serialize the environment state to a dictionary.
        
        Returns:
            Dictionary representation of the state
        """
        with self.lock:
            serialized = {
                "timestamp": self.timestamp.isoformat(),
                "objects": {},
                "robots": {}
            }
            
            # Serialize objects
            for obj_id, obj_state in self.objects.items():
                serialized["objects"][obj_id] = {}
                for key, value in obj_state.items():
                    if isinstance(value, np.ndarray):
                        serialized["objects"][obj_id][key] = value.tolist()
                    elif isinstance(value, datetime):
                        serialized["objects"][obj_id][key] = value.isoformat()
                    else:
                        serialized["objects"][obj_id][key] = value
            
            # Serialize robots
            for robot_id, robot_state in self.robots.items():
                serialized["robots"][robot_id] = {}
                for key, value in robot_state.items():
                    if isinstance(value, np.ndarray):
                        serialized["robots"][robot_id][key] = value.tolist()
                    elif isinstance(value, datetime):
                        serialized["robots"][robot_id][key] = value.isoformat()
                    else:
                        serialized["robots"][robot_id][key] = value
            
            return serialized