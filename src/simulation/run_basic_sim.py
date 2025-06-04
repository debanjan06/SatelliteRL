#!/usr/bin/env python3
"""
Basic Satellite Constellation Simulation
This script demonstrates the foundational simulation capabilities of SatelliteRL.

Author: Debanjan Shil
Date: June 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math

@dataclass
class Satellite:
    """Represents a satellite in the constellation"""
    name: str
    altitude: float  # km
    inclination: float  # degrees
    longitude: float  # degrees (starting position)
    period: float  # orbital period in minutes
    power_level: float = 100.0  # percentage
    data_storage: float = 0.0  # GB used
    max_storage: float = 500.0  # GB capacity
    
class GroundStation:
    """Represents a ground station for data downlink"""
    def __init__(self, name: str, lat: float, lon: float, elevation_mask: float = 10.0):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.elevation_mask = elevation_mask  # minimum elevation angle
        self.contact_windows = []

class ObservationRequest:
    """Represents an Earth observation request"""
    def __init__(self, target_lat: float, target_lon: float, priority: int, 
                 deadline: datetime, data_type: str = "optical"):
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.priority = priority  # 1-5, where 5 is highest
        self.deadline = deadline
        self.data_type = data_type
        self.completed = False
        self.assigned_satellite = None

class SatelliteConstellationSimulator:
    """Main simulation class for satellite constellation management"""
    
    def __init__(self):
        self.satellites = []
        self.ground_stations = []
        self.observation_requests = []
        self.current_time = datetime.now()
        self.simulation_duration = timedelta(hours=24)  # 24-hour simulation
        self.time_step = timedelta(minutes=5)  # 5-minute time steps
        
        # Performance metrics
        self.metrics = {
            'requests_completed': 0,
            'total_requests': 0,
            'power_usage': [],
            'data_collected': [],
            'coverage_efficiency': []
        }
        
    def add_satellite(self, satellite: Satellite):
        """Add a satellite to the constellation"""
        self.satellites.append(satellite)
        
    def add_ground_station(self, station: GroundStation):
        """Add a ground station to the network"""
        self.ground_stations.append(station)
        
    def add_observation_request(self, request: ObservationRequest):
        """Add an observation request to the queue"""
        self.observation_requests.append(request)
        
    def calculate_satellite_position(self, satellite: Satellite, time_offset: float) -> Tuple[float, float]:
        """Calculate satellite position at given time offset (in hours)"""
        # Simplified orbital mechanics - real implementation would use SGP4
        angular_velocity = 360.0 / (satellite.period / 60.0)  # degrees per hour
        current_longitude = (satellite.longitude + angular_velocity * time_offset) % 360
        
        # Simplified ground track - assumes circular orbit
        # In reality, this would be much more complex
        latitude = satellite.inclination * math.sin(math.radians(angular_velocity * time_offset * 4))
        
        return latitude, current_longitude
        
    def check_visibility(self, sat_lat: float, sat_lon: float, target_lat: float, target_lon: float) -> bool:
        """Check if satellite can observe target location"""
        # Simple visibility check based on distance
        # Real implementation would consider elevation angles, atmosphere, etc.
        distance = math.sqrt((sat_lat - target_lat)**2 + (sat_lon - target_lon)**2)
        return distance < 10.0  # Within 10 degrees
        
    def simulate_scheduling_algorithm(self, algorithm_type: str = "greedy"):
        """Simulate different scheduling algorithms"""
        if algorithm_type == "greedy":
            return self._greedy_scheduler()
        elif algorithm_type == "priority":
            return self._priority_scheduler()
        else:
            return self._random_scheduler()
            
    def _greedy_scheduler(self) -> List[Tuple[str, str, float]]:
        """Greedy scheduling: assign first available satellite to each request"""
        assignments = []
        
        for request in self.observation_requests:
            if request.completed:
                continue
                
            best_satellite = None
            best_score = -1
            
            for satellite in self.satellites:
                if satellite.power_level < 20:  # Not enough power
                    continue
                    
                # Calculate when satellite will be over target
                for hour_offset in np.arange(0, 24, 0.5):  # Check every 30 minutes
                    sat_lat, sat_lon = self.calculate_satellite_position(satellite, hour_offset)
                    
                    if self.check_visibility(sat_lat, sat_lon, request.target_lat, request.target_lon):
                        # Score based on power level and time to target
                        score = satellite.power_level - hour_offset
                        if score > best_score:
                            best_score = score
                            best_satellite = satellite
                            break
                            
            if best_satellite:
                assignments.append((request.target_lat, request.target_lon, best_satellite.name, best_score))
                request.assigned_satellite = best_satellite.name
                request.completed = True
                best_satellite.power_level -= 15  # Consume power for observation
                best_satellite.data_storage += np.random.uniform(5, 20)  # Add collected data
                
        return assignments
        
    def _priority_scheduler(self) -> List[Tuple[str, str, float]]:
        """Priority-based scheduling: handle high-priority requests first"""
        # Sort requests by priority (highest first)
        sorted_requests = sorted(self.observation_requests, key=lambda x: x.priority, reverse=True)
        assignments = []
        
        for request in sorted_requests:
            if request.completed:
                continue
                
            # Similar logic to greedy but prioritize high-priority requests
            best_satellite = None
            best_score = -1
            
            for satellite in self.satellites:
                if satellite.power_level < 20:
                    continue
                    
                for hour_offset in np.arange(0, 24, 0.5):
                    sat_lat, sat_lon = self.calculate_satellite_position(satellite, hour_offset)
                    
                    if self.check_visibility(sat_lat, sat_lon, request.target_lat, request.target_lon):
                        # Higher score for higher priority requests
                        score = satellite.power_level - hour_offset + request.priority * 10
                        if score > best_score:
                            best_score = score
                            best_satellite = satellite
                            break
                            
            if best_satellite:
                assignments.append((request.target_lat, request.target_lon, best_satellite.name, best_score))
                request.assigned_satellite = best_satellite.name
                request.completed = True
                best_satellite.power_level -= 15
                best_satellite.data_storage += np.random.uniform(5, 20)
                
        return assignments
        
    def _random_scheduler(self) -> List[Tuple[str, str, float]]:
        """Random scheduling: baseline for comparison"""
        assignments = []
        
        for request in self.observation_requests:
            if request.completed:
                continue
                
            # Randomly select available satellite
            available_sats = [s for s in self.satellites if s.power_level >= 20]
            if available_sats:
                chosen_sat = np.random.choice(available_sats)
                assignments.append((request.target_lat, request.target_lon, chosen_sat.name, 0))
                request.assigned_satellite = chosen_sat.name
                request.completed = True
                chosen_sat.power_level -= 15
                chosen_sat.data_storage += np.random.uniform(5, 20)
                
        return assignments
        
    def run_simulation(self, algorithm_type: str = "greedy") -> Dict:
        """Run the complete simulation"""
        print(f"🚀 Starting SatelliteRL Simulation")
        print(f"📊 Constellation: {len(self.satellites)} satellites")
        print(f"🎯 Observation requests: {len(self.observation_requests)}")
        print(f"🌍 Ground stations: {len(self.ground_stations)}")
        print(f"⚡ Algorithm: {algorithm_type}")
        print("-" * 50)
        
        # Run scheduling algorithm
        assignments = self.simulate_scheduling_algorithm(algorithm_type)
        
        # Calculate metrics
        completed_requests = len([r for r in self.observation_requests if r.completed])
        total_requests = len(self.observation_requests)
        completion_rate = (completed_requests / total_requests) * 100 if total_requests > 0 else 0
        
        avg_power_usage = np.mean([s.power_level for s in self.satellites])
        total_data_collected = sum([s.data_storage for s in self.satellites])
        
        # Store metrics
        self.metrics['requests_completed'] = completed_requests
        self.metrics['total_requests'] = total_requests
        self.metrics['completion_rate'] = completion_rate
        self.metrics['avg_power_usage'] = avg_power_usage
        self.metrics['total_data_collected'] = total_data_collected
        
        # Print results
        print(f"✅ Requests completed: {completed_requests}/{total_requests} ({completion_rate:.1f}%)")
        print(f"⚡ Average power remaining: {avg_power_usage:.1f}%")
        print(f"💾 Total data collected: {total_data_collected:.1f} GB")
        print(f"🎯 Assignments made: {len(assignments)}")
        
        return {
            'assignments': assignments,
            'metrics': self.metrics,
            'satellites': self.satellites,
            'requests': self.observation_requests
        }
        
    def visualize_results(self, results: Dict):
        """Create visualization of simulation results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Completion Rate by Algorithm
        algorithms = ['Greedy', 'Priority', 'Random']
        completion_rates = []
        
        for alg in ['greedy', 'priority', 'random']:
            # Reset and run simulation for each algorithm
            self.reset_simulation()
            temp_results = self.run_simulation(alg.lower())
            completion_rates.append(temp_results['metrics']['completion_rate'])
            
        ax1.bar(algorithms, completion_rates, color=['green', 'blue', 'red'])
        ax1.set_title('Completion Rate by Algorithm')
        ax1.set_ylabel('Completion Rate (%)')
        ax1.set_ylim(0, 100)
        
        # 2. Satellite Power Levels
        sat_names = [s.name for s in self.satellites]
        power_levels = [s.power_level for s in self.satellites]
        colors = ['red' if p < 30 else 'orange' if p < 60 else 'green' for p in power_levels]
        
        ax2.bar(sat_names, power_levels, color=colors)
        ax2.set_title('Satellite Power Levels After Simulation')
        ax2.set_ylabel('Power Level (%)')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Data Storage Usage
        storage_usage = [s.data_storage for s in self.satellites]
        storage_capacity = [s.max_storage for s in self.satellites]
        
        x_pos = np.arange(len(sat_names))
        width = 0.35
        
        ax3.bar(x_pos - width/2, storage_usage, width, label='Used Storage', color='skyblue')
        ax3.bar(x_pos + width/2, storage_capacity, width, label='Total Capacity', color='lightgray', alpha=0.7)
        ax3.set_title('Satellite Data Storage Status')
        ax3.set_ylabel('Storage (GB)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(sat_names, rotation=45)
        ax3.legend()
        
        # 4. Request Priority Distribution
        priorities = [r.priority for r in self.observation_requests]
        completed_priorities = [r.priority for r in self.observation_requests if r.completed]
        
        priority_counts = {i: priorities.count(i) for i in range(1, 6)}
        completed_counts = {i: completed_priorities.count(i) for i in range(1, 6)}
        
        x_pos = list(priority_counts.keys())
        total_bars = list(priority_counts.values())
        completed_bars = [completed_counts.get(i, 0) for i in x_pos]
        
        ax4.bar(x_pos, total_bars, label='Total Requests', color='lightcoral', alpha=0.7)
        ax4.bar(x_pos, completed_bars, label='Completed', color='darkgreen')
        ax4.set_title('Request Completion by Priority Level')
        ax4.set_xlabel('Priority Level')
        ax4.set_ylabel('Number of Requests')
        ax4.set_xticks(x_pos)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('results/simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def reset_simulation(self):
        """Reset simulation state for multiple runs"""
        for satellite in self.satellites:
            satellite.power_level = 100.0
            satellite.data_storage = 0.0
            
        for request in self.observation_requests:
            request.completed = False
            request.assigned_satellite = None

def create_sample_constellation() -> SatelliteConstellationSimulator:
    """Create a sample constellation for demonstration"""
    sim = SatelliteConstellationSimulator()
    
    # Add satellites (simplified constellation similar to Planet Labs)
    satellites = [
        Satellite("DOVE-001", altitude=500, inclination=97.4, longitude=0, period=95),
        Satellite("DOVE-002", altitude=500, inclination=97.4, longitude=45, period=95),
        Satellite("DOVE-003", altitude=500, inclination=97.4, longitude=90, period=95),
        Satellite("DOVE-004", altitude=500, inclination=97.4, longitude=135, period=95),
        Satellite("DOVE-005", altitude=500, inclination=97.4, longitude=180, period=95),
        Satellite("SKYSAT-001", altitude=450, inclination=53, longitude=30, period=93),
        Satellite("SKYSAT-002", altitude=450, inclination=53, longitude=150, period=93),
    ]
    
    for sat in satellites:
        sim.add_satellite(sat)
    
    # Add ground stations
    ground_stations = [
        GroundStation("Svalbard", 78.9, 11.9),  # Arctic station
        GroundStation("McMurdo", -77.8, 166.7),  # Antarctic station
        GroundStation("Fairbanks", 64.8, -147.7),  # Alaska
        GroundStation("Singapore", 1.3, 103.8),  # Equatorial
    ]
    
    for station in ground_stations:
        sim.add_ground_station(station)
    
    # Add observation requests
    requests = [
        # High-priority disaster response
        ObservationRequest(35.0, 139.0, 5, datetime.now() + timedelta(hours=2), "optical"),  # Tokyo area
        ObservationRequest(-33.9, 18.4, 5, datetime.now() + timedelta(hours=1), "thermal"),  # Cape Town
        
        # Medium-priority environmental monitoring
        ObservationRequest(64.1, -21.9, 3, datetime.now() + timedelta(hours=6), "optical"),  # Iceland
        ObservationRequest(-14.2, -51.9, 3, datetime.now() + timedelta(hours=8), "thermal"),  # Amazon
        ObservationRequest(27.2, 77.1, 3, datetime.now() + timedelta(hours=12), "optical"), # Northern India
        
        # Low-priority routine monitoring
        ObservationRequest(40.7, -74.0, 2, datetime.now() + timedelta(hours=18), "optical"), # New York
        ObservationRequest(51.5, -0.1, 2, datetime.now() + timedelta(hours=20), "optical"),  # London
        ObservationRequest(37.8, -122.4, 1, datetime.now() + timedelta(hours=22), "thermal"), # San Francisco
        ObservationRequest(-34.6, -58.4, 1, datetime.now() + timedelta(hours=24), "optical"), # Buenos Aires
        
        # Agricultural monitoring
        ObservationRequest(41.9, -87.6, 2, datetime.now() + timedelta(hours=15), "optical"), # Chicago area
        ObservationRequest(52.5, 13.4, 2, datetime.now() + timedelta(hours=16), "optical"),  # Berlin area
    ]
    
    for request in requests:
        sim.add_observation_request(request)
    
    return sim

def main():
    """Main execution function"""
    print("🛰️  SatelliteRL Foundation Simulation")
    print("=" * 60)
    
    # Create and run simulation
    simulator = create_sample_constellation()
    
    # Run with different algorithms and compare
    algorithms = ['greedy', 'priority', 'random']
    results_comparison = {}
    
    for algorithm in algorithms:
        print(f"\n🔄 Running {algorithm.upper()} algorithm...")
        simulator.reset_simulation()
        results = simulator.run_simulation(algorithm)
        results_comparison[algorithm] = results['metrics']
    
    # Print comparison
    print("\n📊 ALGORITHM COMPARISON")
    print("-" * 60)
    print(f"{'Algorithm':<12} {'Completion %':<15} {'Avg Power':<12} {'Data Collected':<15}")
    print("-" * 60)
    
    for alg, metrics in results_comparison.items():
        print(f"{alg.capitalize():<12} {metrics['completion_rate']:<15.1f} "
              f"{metrics['avg_power_usage']:<12.1f} {metrics['total_data_collected']:<15.1f}")
    
    # Generate visualization
    print("\n📈 Generating visualization...")
    simulator.visualize_results(results)
    
    # Save results
    import json
    import os
    
    os.makedirs('results', exist_ok=True)
    
    with open('results/simulation_metrics.json', 'w') as f:
        json.dump(results_comparison, f, indent=2)
    
    print("\n✅ Simulation completed successfully!")
    print("📁 Results saved to 'results/' directory")
    print("🎯 Next steps:")
    print("   1. Implement RL agents in src/agents/")
    print("   2. Create Gym environment in src/environment/")
    print("   3. Add real orbital mechanics with Skyfield")
    print("   4. Integrate weather data APIs")

if __name__ == "__main__":
    main()