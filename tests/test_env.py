import gymnasium as gym
import pytest
from src.environment.satellite_env import SatelliteEnvironment

def test_env_initialization():
    """Ensure the custom space environment initializes without throwing errors."""
    try:
        # Simple instantiation test to protect against broken imports or paths
        env = gym.make('SatelliteRL-v0') # Or your direct class instantiation
        assert env is not None
    except Exception as e:
        pytest.fail(f"Environment failed to initialize: {e}")