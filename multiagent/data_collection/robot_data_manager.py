"""
Module for managing data collection from multiple robots.
"""
import threading
import queue
import time
from datetime import datetime

class RobotDataManager:
    """
    Manager for collecting and synchronizing data from multiple robots.
    """
    
    def __init__(self, max_queue_size=100):
        """
        Initialize robot data manager.
        
        Args:
            max_queue_size: Maximum size of data queues
        """
        self.robots = {}  # Dictionary mapping robot_id to robot data sources
        self.data_queues = {}  # Data queues for each robot
        self.max_queue_size = max_queue_size
        self.collection_threads = {}
        self.stop_events = {}
        self.lock = threading.Lock()
    
    def add_robot(self, robot_id, data_sources):
        """
        Add a robot with its data sources.
        
        Args:
            robot_id: Unique identifier for the robot
            data_sources: Dictionary mapping source_name to data source object
                         (e.g., {"camera": camera_interface})
        """
        with self.lock:
            if robot_id in self.robots:
                raise ValueError(f"Robot with ID {robot_id} already exists")
            
            self.robots[robot_id] = data_sources
            self.data_queues[robot_id] = {
                source_name: queue.Queue(maxsize=self.max_queue_size)
                for source_name in data_sources
            }
            self.stop_events[robot_id] = threading.Event()
    
    def start_collection(self, robot_id):
        """
        Start data collection for a specific robot.
        
        Args:
            robot_id: Identifier for the robot
        """
        if robot_id not in self.robots:
            raise ValueError(f"Robot with ID {robot_id} not found")
        
        # Stop any existing collection thread
        if robot_id in self.collection_threads and self.collection_threads[robot_id].is_alive():
            self.stop_collection(robot_id)
        
        # Reset stop event
        self.stop_events[robot_id].clear()
        
        # Create and start collection thread
        thread = threading.Thread(
            target=self._collection_worker,
            args=(robot_id,),
            daemon=True
        )
        self.collection_threads[robot_id] = thread
        thread.start()
    
    def stop_collection(self, robot_id):
        """
        Stop data collection for a specific robot.
        
        Args:
            robot_id: Identifier for the robot
        """
        if robot_id in self.stop_events:
            self.stop_events[robot_id].set()
            
            # Wait for thread to finish
            if robot_id in self.collection_threads and self.collection_threads[robot_id].is_alive():
                self.collection_threads[robot_id].join(timeout=1.0)
    
    def _collection_worker(self, robot_id):
        """Worker thread function for collecting data from a robot."""
        data_sources = self.robots[robot_id]
        data_queues = self.data_queues[robot_id]
        stop_event = self.stop_events[robot_id]
        
        # Start all data sources
        for source_name, source in data_sources.items():
            if hasattr(source, 'start_stream'):
                source.start_stream()
        
        # Collect data until stopped
        while not stop_event.is_set():
            for source_name, source in data_sources.items():
                try:
                    # Get data from source
                    if hasattr(source, 'get_frame'):
                        data = source.get_frame()
                    else:
                        # Generic data getter for non-camera sources
                        data = getattr(source, 'get_data', lambda: None)()
                    
                    if data is not None:
                        # Add collection timestamp if not present
                        if isinstance(data, dict) and "timestamp" not in data:
                            data["timestamp"] = datetime.now()
                        
                        # Add to queue, dropping oldest if full
                        queue = data_queues[source_name]
                        try:
                            queue.put_nowait(data)
                        except queue.Full:
                            # Remove oldest item
                            try:
                                queue.get_nowait()
                                queue.put_nowait(data)
                            except (queue.Empty, queue.Full):
                                pass  # Can happen in rare race conditions
                except Exception as e:
                    print(f"Error collecting data from {robot_id}.{source_name}: {e}")
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.01)
        
        # Stop all data sources
        for source_name, source in data_sources.items():
            if hasattr(source, 'stop_stream'):
                source.stop_stream()
    
    def get_latest_data(self, robot_id, source_name):
        """
        Get the latest data from a specific robot's data source.
        
        Args:
            robot_id: Identifier for the robot
            source_name: Name of the data source
            
        Returns:
            Latest data or None if no data available
        """
        if robot_id not in self.data_queues or source_name not in self.data_queues[robot_id]:
            return None
        
        queue = self.data_queues[robot_id][source_name]
        if queue.empty():
            return None
        
        # Get latest data by emptying queue
        latest_data = None
        while not queue.empty():
            try:
                latest_data = queue.get_nowait()
                queue.task_done()
            except queue.Empty:
                break
        
        return latest_data