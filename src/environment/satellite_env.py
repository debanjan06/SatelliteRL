#!/usr/bin/env python3
"""
Satellite Constellation Gym Environment for Reinforcement Learning

This module implements a custom OpenAI Gym environment using Skyfield
for accurate Two-Line Element (TLE) orbital propagation.

Author: Debanjan Shil
Date: June 2026
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from skyfield.api import load, EarthSatellite, wgs84
from datetime import datetime, timedelta

@dataclass
class SatelliteState:
    name: str
    satellite_obj: EarthSatellite
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    power_level: float
    data_storage: float
    max_storage: float
    thermal_state: float
    communication_window: bool
    instrument_health: Dict[str, float]
    
@dataclass 
class ObservationTarget:
    latitude: float
    longitude: float
    priority: int
    deadline: float
    data_type: str
    cloud_coverage: float
    value: float
    
class SatelliteEnvironment(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 num_satellites: int = 5,
                 max_targets: int = 20,
                 observation_horizon: float = 24.0,
                 time_step: float = 0.1,
                 reward_scaling: float = 1.0):
        
        super().__init__()
        
        self.num_satellites = min(num_satellites, 5)
        self.max_targets = max_targets
        self.observation_horizon = observation_horizon
        self.time_step = time_step
        self.reward_scaling = reward_scaling
        
        self.ts = load.timescale()
        self.base_time = self.ts.now()
        
        self.current_time_offset = 0.0
        self.satellites = []
        self.targets = []
        self.completed_observations = []
        self.total_reward = 0.0
        
        self.action_space = spaces.MultiDiscrete([max_targets + 1] * self.num_satellites)
        
        satellite_state_size = 15
        target_state_size = 7
        global_state_size = 5
        
        total_obs_size = (
            self.num_satellites * satellite_state_size +
            max_targets * target_state_size +
            global_state_size
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
        self.episode_stats = {
            'targets_completed': 0,
            'total_value_gained': 0.0,
            'power_efficiency': 0.0,
            'missed_deadlines': 0,
            'emergency_situations': 0
        }

        self.sample_tles = [
            ("PLANET-1", "1 43042U 17066C   23100.50000000  .00001000  00000-0  50000-4 0  9999", "2 43042  97.5000 150.0000 0010000  90.0000 270.0000 15.10000000250001"),
            ("PLANET-2", "1 43043U 17066D   23100.50000000  .00001000  00000-0  50000-4 0  9999", "2 43043  97.5000 160.0000 0010000  90.0000 270.0000 15.10000000250002"),
            ("PLANET-3", "1 43044U 17066E   23100.50000000  .00001000  00000-0  50000-4 0  9999", "2 43044  97.5000 170.0000 0010000  90.0000 270.0000 15.10000000250003"),
            ("PLANET-4", "1 43045U 17066F   23100.50000000  .00001000  00000-0  50000-4 0  9999", "2 43045  97.5000 180.0000 0010000  90.0000 270.0000 15.10000000250004"),
            ("PLANET-5", "1 43046U 17066G   23100.50000000  .00001000  00000-0  50000-4 0  9999", "2 43046  97.5000 190.0000 0010000  90.0000 270.0000 15.10000000250005")
        ]
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            
        self.current_time_offset = 0.0
        self.base_time = self.ts.now()
        
        self.satellites = []
        for i in range(self.num_satellites):
            name, line1, line2 = self.sample_tles[i]
            sat_obj = EarthSatellite(line1, line2, name, self.ts)
            
            geocentric = sat_obj.at(self.base_time)
            subpoint = wgs84.subpoint(geocentric)
            
            initial_pos = (
                subpoint.latitude.degrees,
                subpoint.longitude.degrees,
                subpoint.elevation.km
            )
            
            initial_vel = (7.5, 0.0, 0.0)
            
            satellite = SatelliteState(
                name=name,
                satellite_obj=sat_obj,
                position=initial_pos,
                velocity=initial_vel,
                power_level=np.random.uniform(80, 100),
                data_storage=np.random.uniform(0, 50),
                max_storage=500.0,
                thermal_state=np.random.uniform(15, 25),
                communication_window=np.random.choice([True, False]),
                instrument_health={
                    'optical': np.random.uniform(0.8, 1.0),
                    'thermal': np.random.uniform(0.8, 1.0), 
                    'radar': np.random.uniform(0.8, 1.0),
                    'communication': np.random.uniform(0.8, 1.0)
                }
            )
            self.satellites.append(satellite)
            
        self.targets = []
        num_initial_targets = np.random.randint(5, self.max_targets)
        for _ in range(num_initial_targets):
            self.targets.append(self._generate_observation_target())
            
        self.completed_observations = []
        self.total_reward = 0.0
        self.episode_stats = {
            'targets_completed': 0,
            'total_value_gained': 0.0,
            'power_efficiency': 0.0,
            'missed_deadlines': 0,
            'emergency_situations': 0
        }
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward = 0.0
        
        for sat_idx, target_idx in enumerate(action):
            if target_idx > 0 and target_idx <= len(self.targets):
                target = self.targets[target_idx - 1]
                satellite = self.satellites[sat_idx]
                
                obs_reward, obs_success = self._attempt_observation(satellite, target)
                reward += obs_reward
                
                if obs_success:
                    self.completed_observations.append({
                        'satellite': sat_idx,
                        'target': target,
                        'time': self.current_time_offset,
                        'value': target.value
                    })
                    self.episode_stats['targets_completed'] += 1
                    self.episode_stats['total_value_gained'] += target.value
        
        self.current_time_offset += self.time_step
        self._update_satellite_states()
        self._update_targets()
        
        reward += self._calculate_global_rewards()
        
        terminated = self.current_time_offset >= self.observation_horizon
        truncated = False
        
        scaled_reward = reward * self.reward_scaling
        self.total_reward += scaled_reward
        
        return self._get_observation(), scaled_reward, terminated, truncated, self._get_info()
    
    def _generate_observation_target(self) -> ObservationTarget:
        return ObservationTarget(
            latitude=np.random.uniform(-50, 50),
            longitude=np.random.uniform(-180, 180),
            priority=np.random.randint(1, 6),
            deadline=np.random.uniform(1, 48),
            data_type=np.random.choice(['optical', 'thermal', 'radar']),
            cloud_coverage=np.random.beta(2, 5),
            value=np.random.uniform(10, 100)
        )
    
    def _attempt_observation(self, satellite: SatelliteState, target: ObservationTarget) -> Tuple[float, bool]:
        can_observe, visibility_factor = self._check_observation_feasibility(satellite, target)
        
        if not can_observe:
            return 0.0, False  
        
        quality_factors = {
            'visibility': visibility_factor,
            'power': min(satellite.power_level / 100.0, 1.0),
            'instrument_health': satellite.instrument_health.get(target.data_type, 0.5),
            'cloud_coverage': 1.0 - target.cloud_coverage,
            'deadline_pressure': max(0.1, 1.0 - (self.current_time_offset / target.deadline))
        }
        
        quality = np.mean(list(quality_factors.values()))
        
        base_reward = (target.value * target.priority * quality) * 10.0 
        priority_bonus = (target.priority - 1) * 20.0
        time_bonus = max(0, target.deadline - self.current_time_offset) * 2.0
        
        total_reward = base_reward + priority_bonus + time_bonus
        
        power_consumption = 5.0 * quality if target.data_type == 'optical' else 10.0 * quality
        data_generated = 2.0 * quality
        
        satellite.power_level -= power_consumption
        satellite.data_storage += data_generated
        
        if satellite.power_level < 0:
            satellite.power_level = 0
            total_reward -= 50  
            self.episode_stats['emergency_situations'] += 1
            
        if satellite.data_storage > satellite.max_storage:
            satellite.data_storage = satellite.max_storage
            total_reward -= 10
        
        return total_reward, True
    
    def _check_observation_feasibility(self, satellite: SatelliteState, target: ObservationTarget) -> Tuple[bool, float]:
        if satellite.power_level < 20.0 or satellite.data_storage >= satellite.max_storage * 0.95:
            return False, 0.0
            
        if satellite.instrument_health.get(target.data_type, 0) < 0.3:
            return False, 0.0
            
        sat_lat, sat_lon, sat_alt = satellite.position
        
        lat1, lon1, lat2, lon2 = map(np.radians, [sat_lat, sat_lon, target.latitude, target.longitude])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = c * 6371.0 
        
        max_distance = sat_alt * np.tan(np.radians(30)) 
        
        if distance > max_distance:
            return False, 0.0
            
        visibility_factor = max(0.1, 1.0 - (distance / max_distance))
        return True, visibility_factor
    
    def _update_satellite_states(self):
        current_dt = self.base_time.utc_datetime() + timedelta(hours=self.current_time_offset)
        sim_time = self.ts.utc(current_dt.year, current_dt.month, current_dt.day, 
                               current_dt.hour, current_dt.minute, current_dt.second)
        
        for satellite in self.satellites:
            geocentric = satellite.satellite_obj.at(sim_time)
            subpoint = wgs84.subpoint(geocentric)
            
            satellite.position = (
                subpoint.latitude.degrees,
                subpoint.longitude.degrees,
                subpoint.elevation.km
            )
            
            solar_factor = max(0.1, abs(np.cos(np.radians(satellite.position[0]))))
            power_regen = 3.0 * solar_factor * self.time_step
            satellite.power_level = min(100.0, satellite.power_level + power_regen)
            
            satellite.thermal_state += np.random.normal(0, 0.5)
            satellite.thermal_state = np.clip(satellite.thermal_state, -20, 60)
            satellite.communication_window = np.random.choice([True, False], p=[0.3, 0.7])
    
    def _update_targets(self):
        active_targets = []
        for target in self.targets:
            if self.current_time_offset < target.deadline:
                active_targets.append(target)
            else:
                self.episode_stats['missed_deadlines'] += 1
                
        self.targets = active_targets
        
        if np.random.random() < 0.1 and len(self.targets) < self.max_targets:
            new_target = self._generate_observation_target()
            new_target.deadline += self.current_time_offset
            self.targets.append(new_target)
    
    def _calculate_global_rewards(self) -> float:
        reward = 0.0
        
        avg_power = np.mean([sat.power_level for sat in self.satellites])
        if avg_power < 20:
            reward -= 5.0  
            
        avg_storage_usage = np.mean([sat.data_storage / sat.max_storage for sat in self.satellites])
        if avg_storage_usage > 0.95: 
            reward -= 5.0
            
        emergency_count = sum(1 for sat in self.satellites if sat.power_level < 10)
        reward -= emergency_count * 15
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        obs_components = []
        
        for satellite in self.satellites:
            sat_obs = [
                satellite.position[0] / 90.0,
                satellite.position[1] / 180.0,
                satellite.position[2] / 1000.0,
                satellite.velocity[0] / 10.0,
                satellite.velocity[1] / 10.0,
                satellite.velocity[2] / 10.0,
                satellite.power_level / 100.0,
                satellite.data_storage / satellite.max_storage,
                satellite.thermal_state / 60.0,
                float(satellite.communication_window),
                satellite.instrument_health.get('optical', 0),
                satellite.instrument_health.get('thermal', 0),
                satellite.instrument_health.get('radar', 0),
                satellite.instrument_health.get('communication', 0)
            ]
            obs_components.extend(sat_obs)
        
        while len(obs_components) < self.num_satellites * 15:
            obs_components.extend([0.0] * 15)
            
        for i in range(self.max_targets):
            if i < len(self.targets):
                target = self.targets[i]
                target_obs = [
                    target.latitude / 90.0,
                    target.longitude / 180.0,
                    target.priority / 5.0,
                    min(target.deadline - self.current_time_offset, 48) / 48.0,
                    {'optical': 0, 'thermal': 1, 'radar': 2}.get(target.data_type, 0) / 2.0,
                    target.cloud_coverage,
                    target.value / 100.0
                ]
            else:
                target_obs = [0.0] * 7
                
            obs_components.extend(target_obs)
        
        global_obs = [
            self.current_time_offset / self.observation_horizon,
            len(self.completed_observations) / max(1, len(self.targets) + len(self.completed_observations)),
            np.mean([sat.power_level for sat in self.satellites]) / 100.0,
            0.5, 
            float(any(sat.power_level < 15 for sat in self.satellites))
        ]
        obs_components.extend(global_obs)
        
        return np.array(obs_components, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        return {
            'episode_stats': self.episode_stats.copy(),
            'current_time': self.current_time_offset,
            'active_targets': len(self.targets),
            'completed_observations': len(self.completed_observations),
            'total_reward': self.total_reward
        }
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"\nTime Offset: {self.current_time_offset:.1f} hours")
            print(f"Active Targets: {len(self.targets)}")
            print(f"Completed Observations: {len(self.completed_observations)}")
            for i, sat in enumerate(self.satellites):
                lat, lon, alt = sat.position
                print(f"Sat-{i} [{sat.name}]: Pos=({lat:.1f}, {lon:.1f}, {alt:.0f}km) Power={sat.power_level:.1f}%")