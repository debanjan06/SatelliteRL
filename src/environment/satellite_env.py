#!/usr/bin/env python3
"""
Satellite Constellation Gym Environment for Reinforcement Learning

This module implements a custom OpenAI Gym environment for training RL agents
to manage satellite constellation operations and Earth observation scheduling.

Author: Debanjan Shil
Date: June 2025
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

@dataclass
class SatelliteState:
    """Represents the state of a single satellite"""
    position: Tuple[float, float, float]  # lat, lon, altitude
    velocity: Tuple[float, float, float]  # velocity vector
    power_level: float  # 0-100 percentage
    data_storage: float  # GB used
    max_storage: float  # GB capacity
    thermal_state: float  # temperature for thermal management
    communication_window: bool  # can communicate with ground
    instrument_health: Dict[str, float]  # health of each instrument
    
@dataclass 
class ObservationTarget:
    """Represents an Earth observation target"""
    latitude: float
    longitude: float
    priority: int  # 1-5 scale
    deadline: float  # hours from current time
    data_type: str  # 'optical', 'thermal', 'radar'
    cloud_coverage: float  # 0-1 scale
    value: float  # scientific/commercial value
    
class SatelliteEnvironment(gym.Env):
    """
    Custom Gym environment for satellite constellation management.
    
    The environment simulates a constellation of Earth observation satellites
    that must be scheduled to complete observation requests while managing
    power, storage, and orbital constraints.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 num_satellites: int = 5,
                 max_targets: int = 20,
                 observation_horizon: float = 24.0,  # hours
                 time_step: float = 0.1,  # hours (6 minutes)
                 reward_scaling: float = 1.0):
        
        super().__init__()
        
        # Environment parameters
        self.num_satellites = num_satellites
        self.max_targets = max_targets
        self.observation_horizon = observation_horizon
        self.time_step = time_step
        self.reward_scaling = reward_scaling
        
        # Simulation state
        self.current_time = 0.0  # simulation time in hours
        self.satellites = []
        self.targets = []
        self.completed_observations = []
        self.total_reward = 0.0
        
        # Action space: For each satellite, choose target (0 = no action, 1-max_targets = target index)
        self.action_space = spaces.MultiDiscrete([max_targets + 1] * num_satellites)
        
        # Observation space: satellite states + target states + global info
        satellite_state_size = 15  # position(3) + velocity(3) + power(1) + storage(2) + thermal(1) + comm(1) + instruments(4)
        target_state_size = 7      # lat(1) + lon(1) + priority(1) + deadline(1) + type(1) + clouds(1) + value(1)
        global_state_size = 5      # current_time(1) + total_completed(1) + avg_power(1) + weather_index(1) + emergency_flag(1)
        
        total_obs_size = (
            num_satellites * satellite_state_size +  # satellite states
            max_targets * target_state_size +        # target states  
            global_state_size                        # global state
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
        # Performance tracking
        self.episode_stats = {
            'targets_completed': 0,
            'total_value_gained': 0.0,
            'power_efficiency': 0.0,
            'missed_deadlines': 0,
            'emergency_situations': 0
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            
        # Reset simulation time
        self.current_time = 0.0
        
        # Initialize satellites with random orbital positions
        self.satellites = []
        for i in range(self.num_satellites):
            satellite = SatelliteState(
                position=self._generate_satellite_position(),
                velocity=self._generate_satellite_velocity(),
                power_level=np.random.uniform(80, 100),  # Start with high power
                data_storage=np.random.uniform(0, 50),   # Some initial data
                max_storage=500.0,
                thermal_state=np.random.uniform(15, 25), # Normal operating temp
                communication_window=np.random.choice([True, False]),
                instrument_health={
                    'optical': np.random.uniform(0.8, 1.0),
                    'thermal': np.random.uniform(0.8, 1.0), 
                    'radar': np.random.uniform(0.8, 1.0),
                    'communication': np.random.uniform(0.8, 1.0)
                }
            )
            self.satellites.append(satellite)
            
        # Generate initial observation targets
        self.targets = []
        num_initial_targets = np.random.randint(5, self.max_targets)
        for _ in range(num_initial_targets):
            target = self._generate_observation_target()
            self.targets.append(target)
            
        # Reset tracking variables
        self.completed_observations = []
        self.total_reward = 0.0
        self.episode_stats = {
            'targets_completed': 0,
            'total_value_gained': 0.0,
            'power_efficiency': 0.0,
            'missed_deadlines': 0,
            'emergency_situations': 0
        }
        
        # Return initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        
        # Process actions for each satellite
        reward = 0.0
        info = {}
        
        for sat_idx, target_idx in enumerate(action):
            if target_idx > 0 and target_idx <= len(self.targets):
                # Satellite is attempting to observe target
                target = self.targets[target_idx - 1]
                satellite = self.satellites[sat_idx]
                
                # Check if observation is possible
                obs_reward, obs_success = self._attempt_observation(satellite, target, sat_idx)
                reward += obs_reward
                
                if obs_success:
                    # Remove completed target or mark as completed
                    self.completed_observations.append({
                        'satellite': sat_idx,
                        'target': target,
                        'time': self.current_time,
                        'value': target.value
                    })
                    self.episode_stats['targets_completed'] += 1
                    self.episode_stats['total_value_gained'] += target.value
        
        # Update satellite states (orbital mechanics, power consumption, etc.)
        self._update_satellite_states()
        
        # Add/remove targets dynamically
        self._update_targets()
        
        # Calculate additional rewards/penalties
        reward += self._calculate_global_rewards()
        
        # Update simulation time
        self.current_time += self.time_step
        
        # Check termination conditions
        terminated = self.current_time >= self.observation_horizon
        truncated = False  # Could add other truncation conditions
        
        # Calculate final reward scaling
        scaled_reward = reward * self.reward_scaling
        self.total_reward += scaled_reward
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, scaled_reward, terminated, truncated, info
    
    def _generate_satellite_position(self) -> Tuple[float, float, float]:
        """Generate realistic satellite orbital position"""
        # Simplified: random position in LEO
        latitude = np.random.uniform(-90, 90)
        longitude = np.random.uniform(-180, 180)
        altitude = np.random.uniform(400, 600)  # LEO altitude in km
        return (latitude, longitude, altitude)
    
    def _generate_satellite_velocity(self) -> Tuple[float, float, float]:
        """Generate satellite velocity vector"""
        # Simplified orbital velocity calculation
        velocity_magnitude = 7.5  # km/s for LEO
        # Random direction components (simplified)
        vx = np.random.uniform(-velocity_magnitude, velocity_magnitude)
        vy = np.random.uniform(-velocity_magnitude, velocity_magnitude) 
        vz = np.random.uniform(-1, 1)  # Small vertical component
        return (vx, vy, vz)
    
    def _generate_observation_target(self) -> ObservationTarget:
        """Generate a new observation target"""
        target = ObservationTarget(
            latitude=np.random.uniform(-60, 60),  # Avoid polar regions mostly
            longitude=np.random.uniform(-180, 180),
            priority=np.random.randint(1, 6),
            deadline=np.random.uniform(1, 48),  # 1-48 hours deadline
            data_type=np.random.choice(['optical', 'thermal', 'radar']),
            cloud_coverage=np.random.beta(2, 5),  # Biased toward low cloud cover
            value=np.random.uniform(10, 100)
        )
        return target
    
    def _attempt_observation(self, satellite: SatelliteState, target: ObservationTarget, sat_idx: int) -> Tuple[float, bool]:
        """Attempt to perform observation and return reward and success status"""
        
        # Check if satellite can observe target (visibility, power, etc.)
        can_observe, visibility_factor = self._check_observation_feasibility(satellite, target)
        
        if not can_observe:
            return -1.0, False  # Small penalty for failed attempt
        
        # Calculate observation quality and reward
        quality_factors = {
            'visibility': visibility_factor,
            'power': min(satellite.power_level / 100.0, 1.0),
            'instrument_health': satellite.instrument_health.get(target.data_type, 0.5),
            'cloud_coverage': 1.0 - target.cloud_coverage,
            'deadline_pressure': max(0.1, 1.0 - (self.current_time / target.deadline))
        }
        
        # Overall quality score
        quality = np.mean(list(quality_factors.values()))
        
        # Base reward from target value and priority
        base_reward = target.value * target.priority * quality
        
        # Bonus for high-priority targets
        priority_bonus = (target.priority - 1) * 5
        
        # Time bonus (reward completing earlier)
        time_bonus = max(0, target.deadline - self.current_time) * 0.5
        
        total_reward = base_reward + priority_bonus + time_bonus
        
        # Update satellite state after observation
        power_consumption = self._calculate_power_consumption(target.data_type, quality)
        data_generated = self._calculate_data_volume(target.data_type, quality)
        
        satellite.power_level -= power_consumption
        satellite.data_storage += data_generated
        
        # Check for constraint violations
        if satellite.power_level < 0:
            satellite.power_level = 0
            total_reward -= 20  # Penalty for power depletion
            self.episode_stats['emergency_situations'] += 1
            
        if satellite.data_storage > satellite.max_storage:
            satellite.data_storage = satellite.max_storage
            total_reward -= 10  # Penalty for storage overflow
        
        return total_reward, True
    
    def _check_observation_feasibility(self, satellite: SatelliteState, target: ObservationTarget) -> Tuple[bool, float]:
        """Check if satellite can observe target and return visibility factor"""
        
        # Check power requirements
        min_power_required = 20.0
        if satellite.power_level < min_power_required:
            return False, 0.0
        
        # Check storage capacity
        if satellite.data_storage >= satellite.max_storage * 0.95:
            return False, 0.0
        
        # Check instrument health
        required_instrument = target.data_type
        if satellite.instrument_health.get(required_instrument, 0) < 0.3:
            return False, 0.0
        
        # Calculate visibility based on position (simplified)
        sat_lat, sat_lon, sat_alt = satellite.position
        distance = self._calculate_ground_distance(sat_lat, sat_lon, target.latitude, target.longitude)
        
        # Maximum observation distance based on altitude
        max_distance = sat_alt * 0.1  # Simplified calculation
        
        if distance > max_distance:
            return False, 0.0
        
        # Visibility factor decreases with distance
        visibility_factor = max(0.1, 1.0 - (distance / max_distance))
        
        return True, visibility_factor
    
    def _calculate_ground_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points on Earth"""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth radius in km
        
        return c * r
    
    def _calculate_power_consumption(self, data_type: str, quality: float) -> float:
        """Calculate power consumption for observation"""
        base_consumption = {
            'optical': 5.0,
            'thermal': 8.0,
            'radar': 15.0
        }
        return base_consumption.get(data_type, 5.0) * quality
    
    def _calculate_data_volume(self, data_type: str, quality: float) -> float:
        """Calculate data volume generated by observation"""
        base_volume = {
            'optical': 2.0,  # GB
            'thermal': 1.5,
            'radar': 5.0
        }
        return base_volume.get(data_type, 2.0) * quality
    
    def _update_satellite_states(self):
        """Update satellite orbital positions, power, thermal state, etc."""
        for satellite in self.satellites:
            # Update orbital position (simplified)
            lat, lon, alt = satellite.position
            vx, vy, vz = satellite.velocity
            
            # Simple orbital mechanics update
            angular_velocity = 0.25  # degrees per time step (simplified)
            new_lon = (lon + angular_velocity) % 360
            if new_lon > 180:
                new_lon -= 360
                
            satellite.position = (lat, new_lon, alt)
            
            # Power regeneration from solar panels (simplified)
            solar_factor = max(0.1, abs(math.cos(math.radians(lat))))  # Higher at equator
            power_regen = 3.0 * solar_factor * self.time_step
            satellite.power_level = min(100.0, satellite.power_level + power_regen)
            
            # Thermal management
            satellite.thermal_state += np.random.normal(0, 0.5)  # Random thermal fluctuation
            satellite.thermal_state = np.clip(satellite.thermal_state, -20, 60)
            
            # Communication window updates
            satellite.communication_window = np.random.choice([True, False], p=[0.3, 0.7])
            
            # Instrument degradation (very slow)
            for instrument in satellite.instrument_health:
                degradation = np.random.uniform(0, 0.001) * self.time_step
                satellite.instrument_health[instrument] = max(0, 
                    satellite.instrument_health[instrument] - degradation)
    
    def _update_targets(self):
        """Add new targets and remove expired ones"""
        # Remove expired targets
        active_targets = []
        for target in self.targets:
            if self.current_time < target.deadline:
                active_targets.append(target)
            else:
                self.episode_stats['missed_deadlines'] += 1
                
        self.targets = active_targets
        
        # Add new targets occasionally
        if np.random.random() < 0.1 and len(self.targets) < self.max_targets:  # 10% chance per step
            new_target = self._generate_observation_target()
            new_target.deadline += self.current_time  # Adjust deadline to current time
            self.targets.append(new_target)
    
    def _calculate_global_rewards(self) -> float:
        """Calculate rewards/penalties based on global system state"""
        reward = 0.0
        
        # Power efficiency reward
        avg_power = np.mean([sat.power_level for sat in self.satellites])
        if avg_power > 50:
            reward += 1.0  # Bonus for maintaining good power levels
        elif avg_power < 20:
            reward -= 5.0  # Penalty for low power across constellation
            
        # Storage management
        avg_storage_usage = np.mean([sat.data_storage / sat.max_storage for sat in self.satellites])
        if 0.3 < avg_storage_usage < 0.8:  # Good storage utilization
            reward += 0.5
        elif avg_storage_usage > 0.95:  # Storage nearly full
            reward -= 2.0
            
        # Emergency situations penalty
        emergency_count = sum(1 for sat in self.satellites if sat.power_level < 10)
        reward -= emergency_count * 10
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation"""
        obs_components = []
        
        # Satellite states
        for satellite in self.satellites:
            sat_obs = [
                # Position (3)
                satellite.position[0] / 90.0,   # Normalize latitude
                satellite.position[1] / 180.0,  # Normalize longitude  
                satellite.position[2] / 1000.0, # Normalize altitude
                
                # Velocity (3) 
                satellite.velocity[0] / 10.0,
                satellite.velocity[1] / 10.0,
                satellite.velocity[2] / 10.0,
                
                # Power and storage (3)
                satellite.power_level / 100.0,
                satellite.data_storage / satellite.max_storage,
                satellite.thermal_state / 60.0,
                
                # Communication and instruments (5)
                float(satellite.communication_window),
                satellite.instrument_health.get('optical', 0),
                satellite.instrument_health.get('thermal', 0),
                satellite.instrument_health.get('radar', 0),
                satellite.instrument_health.get('communication', 0)
            ]
            obs_components.extend(sat_obs)
        
        # Pad if fewer satellites than max
        while len(obs_components) < self.num_satellites * 15:
            obs_components.extend([0.0] * 15)
            
        # Target states
        for i in range(self.max_targets):
            if i < len(self.targets):
                target = self.targets[i]
                target_obs = [
                    target.latitude / 90.0,
                    target.longitude / 180.0,
                    target.priority / 5.0,
                    min(target.deadline - self.current_time, 48) / 48.0,  # Normalize remaining time
                    {'optical': 0, 'thermal': 1, 'radar': 2}.get(target.data_type, 0) / 2.0,
                    target.cloud_coverage,
                    target.value / 100.0
                ]
            else:
                target_obs = [0.0] * 7  # Padding for empty target slots
                
            obs_components.extend(target_obs)
        
        # Global state
        global_obs = [
            self.current_time / self.observation_horizon,
            len(self.completed_observations) / max(1, len(self.targets) + len(self.completed_observations)),
            np.mean([sat.power_level for sat in self.satellites]) / 100.0,
            np.random.uniform(0, 1),  # Weather index placeholder
            float(any(sat.power_level < 15 for sat in self.satellites))  # Emergency flag
        ]
        obs_components.extend(global_obs)
        
        return np.array(obs_components, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information about environment state"""
        return {
            'episode_stats': self.episode_stats.copy(),
            'current_time': self.current_time,
            'active_targets': len(self.targets),
            'completed_observations': len(self.completed_observations),
            'total_reward': self.total_reward,
            'avg_satellite_power': np.mean([sat.power_level for sat in self.satellites]),
            'constellation_health': np.mean([
                np.mean(list(sat.instrument_health.values())) 
                for sat in self.satellites
            ])
        }
    
    def render(self, mode='human'):
        """Render the environment state"""
        if mode == 'human':
            print(f"\n=== Satellite Constellation Status (t={self.current_time:.1f}h) ===")
            print(f"Active Targets: {len(self.targets)}")
            print(f"Completed Observations: {len(self.completed_observations)}")
            print(f"Total Reward: {self.total_reward:.2f}")
            
            print("\nSatellite Status:")
            for i, sat in enumerate(self.satellites):
                lat, lon, alt = sat.position
                print(f"  Sat-{i}: Pos=({lat:.1f}°, {lon:.1f}°, {alt:.0f}km) "
                      f"Power={sat.power_level:.1f}% Storage={sat.data_storage:.1f}GB")
            
            print("\nActive Targets:")
            for i, target in enumerate(self.targets):
                time_left = target.deadline - self.current_time
                print(f"  Target-{i}: ({target.latitude:.1f}°, {target.longitude:.1f}°) "
                      f"Priority={target.priority} TimeLeft={time_left:.1f}h "
                      f"Type={target.data_type}")
        
        elif mode == 'rgb_array':
            # Could implement visual rendering here
            # For now, return a placeholder
            return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up environment resources"""
        pass

# Test function for the environment
def test_satellite_environment():
    """Test the satellite environment with random actions"""
    print("🧪 Testing SatelliteEnvironment...")
    
    env = SatelliteEnvironment(num_satellites=3, max_targets=10)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few steps with random actions
    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            env.render()
        
        if terminated or truncated:
            break
    
    print(f"\n✅ Test completed after {step+1} steps")
    print(f"📊 Total reward: {total_reward:.2f}")
    print(f"🎯 Targets completed: {info['episode_stats']['targets_completed']}")
    
    env.close()

if __name__ == "__main__":
    test_satellite_environment()