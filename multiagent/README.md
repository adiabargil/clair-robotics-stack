# Multi-Agent State Estimation Framework

This module extends the CLAIR robotics stack with capabilities for multi-agent state estimation. It enables multiple robots to share sensor data and maintain a collaborative representation of the environment.

## Architecture

The framework consists of two main components:

1. **Data Collection & Processing**: Handles camera feeds and sensor data collection from multiple robots.
2. **State Estimation**: Fuses data from multiple sources to maintain a shared environment representation.

## Usage Example

```python
# Example of setting up the multi-agent state estimation framework
from multiagent.data_collection.camera_interface import RealCamera
from multiagent.data_collection.image_processor import ImageProcessor
from multiagent.data_collection.robot_data_manager import RobotDataManager
from multiagent.state_estimation.state_representation import EnvironmentState
from multiagent.fusion.data_fusion import DataFusionEngine
from multiagent.state_estimation.state_manager import StateEstimationManager

# Create environment state and fusion engine
env_state = EnvironmentState()
fusion_engine = DataFusionEngine(env_state)

# Create state manager
state_manager = StateEstimationManager(env_state, fusion_engine)

# Set up cameras for robots
robot1_camera = RealCamera(camera_id=0)
robot2_camera = RealCamera(camera_id=1)

# Set up image processors
image_processor = ImageProcessor(resize_dims=(640, 480))

# Create robot data manager
robot_data_manager = RobotDataManager()

# Add robots with their data sources
robot_data_manager.add_robot("robot1", {"camera": robot1_camera})
robot_data_manager.add_robot("robot2", {"camera": robot2_camera})

# Start data collection
robot_data_manager.start_collection("robot1")
robot_data_manager.start_collection("robot2")

# Connect to state manager
state_manager.add_robot_data_source("robot1", robot_data_manager)
state_manager.add_robot_data_source("robot2", robot_data_manager)

# Start state estimation
state_manager.start()

# Define a callback for state updates
def state_update_callback(state):
    print(f"State updated: {len(state['objects'])} objects, {len(state['robots'])} robots")

# Subscribe to state updates
state_manager.subscribe(state_update_callback)

# Application loop
try:
    while True:
        # Get current state
        current_state = state_manager.get_current_state()
        # Use the state for robot control, visualization, etc.
        time.sleep(1)
except KeyboardInterrupt:
    # Clean shutdown
    state_manager.stop()
    robot_data_manager.stop_collection("robot1")
    robot_data_manager.stop_collection("robot2")