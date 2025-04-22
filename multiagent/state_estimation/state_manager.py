"""
Module for managing the state estimation process across multiple robots.
"""
import threading
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StateEstimationManager:
    """
    Manager for the state estimation process in a multi-agent system.
    """
    
    def __init__(self, environment_state, data_fusion_engine, update_interval=0.1):
        """
        Initialize state estimation manager.
        
        Args:
            environment_state: EnvironmentState instance
            data_fusion_engine: DataFusionEngine instance
            update_interval: Time between state updates in seconds
        """
        self.environment_state = environment_state
        self.data_fusion_engine = data_fusion_engine
        self.update_interval = update_interval
        self.running = False
        self.update_thread = None
        self.subscribers = []  # Callbacks to notify on state updates
        self.robot_data_sources = {}  # Data sources for each robot
    
    def add_robot_data_source(self, robot_id, data_source):
        """
        Add a data source for a robot.
        
        Args:
            robot_id: ID of the robot
            data_source: Object with a get_latest_data() method
        """
        self.robot_data_sources[robot_id] = data_source
    
    def start(self):
        """Start the state estimation process."""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("State estimation manager started")
    
    def stop(self):
        """Stop the state estimation process."""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
        logger.info("State estimation manager stopped")
    
    def _update_loop(self):
        """Main loop for updating the state estimate."""
        while self.running:
            try:
                update_start = time.time()
                
                # Collect latest data from all robots
                for robot_id, data_source in self.robot_data_sources.items():
                    try:
                        latest_data = data_source.get_latest_data()
                        if latest_data:
                            # Add observation to fusion engine
                            self.data_fusion_engine.add_observation(robot_id, latest_data)
                    except Exception as e:
                        logger.error(f"Error collecting data from robot {robot_id}: {e}")
                
                # Perform fusion to update the environment state
                self.data_fusion_engine.fuse_robot_positions()
                
                # Notify subscribers
                self._notify_subscribers()
                
                # Calculate sleep time to maintain update frequency
                elapsed = time.time() - update_start
                sleep_time = max(0, self.update_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in state estimation update loop: {e}")
                time.sleep(0.1)  # Sleep to avoid tight error loops
    
    def subscribe(self, callback):
        """
        Subscribe to state updates.
        
        Args:
            callback: Function to call with the updated state
        """
        if callback not in self.subscribers:
            self.subscribers.append(callback)
    
    def unsubscribe(self, callback):
        """
        Unsubscribe from state updates.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _notify_subscribers(self):
        """Notify all subscribers with the current state."""
        current_state = self.environment_state.serialize()
        for callback in self.subscribers:
            try:
                callback(current_state)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}")
    
    def get_current_state(self):
        """
        Get the current environment state.
        
        Returns:
            Serialized environment state
        """
        return self.environment_state.serialize()