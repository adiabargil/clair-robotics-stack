"""
Tests for the state estimation components.
"""
import unittest
import numpy as np
from datetime import datetime, timedelta
from multiagent.state_estimation.state_representation import EnvironmentState
from multiagent.fusion.data_fusion import DataFusionEngine

class TestStateEstimation(unittest.TestCase):
    """Test cases for state estimation components."""
    
    def setUp(self):
        """Set up test environment."""
        self.env_state = EnvironmentState()
        self.fusion_engine = DataFusionEngine(self.env_state)
    
    def test_environment_state(self):
        """Test environment state functionality."""
        # Update object state
        self.env_state.update_object(
            "obj1", 
            position=[1.0, 2.0, 3.0],
            orientation=[1.0, 0.0, 0.0, 0.0],
            properties={"color": "red", "size": 0.5}
        )
        
        # Get object state
        obj_state = self.env_state.get_object("obj1")
        
        # Check values
        self.assertIsNotNone(obj_state)
        np.testing.assert_array_equal(obj_state["position"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(obj_state["orientation"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(obj_state["properties"]["color"], "red")
        self.assertEqual(obj_state["properties"]["size"], 0.5)
    
    def test_data_fusion(self):
        """Test data fusion functionality."""
        # Create observations from two robots
        robot1_observation = {
            "objects": {
                "obj1": {
                    "position": [1.0, 2.0, 3.0],
                    "confidence": 0.8
                }
            },
            "timestamp": datetime.now()
        }
        
        robot2_observation = {
            "objects": {
                "obj1": {
                    "position": [1.1, 2.1, 3.1],
                    "confidence": 0.7
                }
            },
            "timestamp": datetime.now() + timedelta(seconds=1)  # Later observation
        }
        
        # Add observations
        self.fusion_engine.add_observation("robot1", robot1_observation)
        self.fusion_engine.add_observation("robot2", robot2_observation)
        
        # Get object state after fusion
        obj_state = self.env_state.get_object("obj1")
        
        # Should use the more recent observation (robot2)
        self.assertIsNotNone(obj_state)
        np.testing.assert_array_equal(obj_state["position"], [1.1, 2.1, 3.1])

if __name__ == "__main__":
    unittest.main()