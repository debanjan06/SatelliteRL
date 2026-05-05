#!/usr/bin/env python3
"""
Satellite Constellation Simulation Benchmark

This script runs a head-to-head comparison between heuristic baselines
(Random, Greedy) and the trained Multi-Agent DQN. It outputs the final
metrics to simulation_metrics.json for analysis.

Author: Debanjan Shil
Date: June 2026
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import glob

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.satellite_env import SatelliteEnvironment
from agents.dqn_agent import DQNAgent

def load_config(config_path="configs/default.yml"):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}. Using safe defaults.")
        return {
            'environment': {
                'num_satellites': 5,
                'max_targets': 20,
                'observation_horizon': 24.0,
                'time_step': 0.1
            }
        }

def find_latest_model():
    """Finds the most recently saved DQN model"""
    model_files = glob.glob('models/final_dqn_*.pth')
    if not model_files:
        # Fallback to the best checkpoints if final doesn't exist
        model_files = glob.glob('models/best_models/*.pth')
    
    if not model_files:
        return None
        
    return max(model_files, key=os.path.getctime)

def run_simulation(env_config, agent_type="random", model_path=None, seed=42):
    """Runs a single simulation episode and collects metrics"""
    print(f"\n🚀 Running Simulation: {agent_type.upper()}")
    
    env = SatelliteEnvironment(
        num_satellites=env_config['num_satellites'],
        max_targets=env_config['max_targets'],
        observation_horizon=env_config['observation_horizon'],
        time_step=env_config['time_step']
    )
    
    state, _ = env.reset(seed=seed)
    
    # Initialize DQN agent if required
    agent = None
    if agent_type == "dqn":
        if model_path is None:
            raise FileNotFoundError("Model path required for DQN simulation.")
            
        actual_state_size = len(state)
        action_size = env_config['max_targets'] + 1
        
        agent = DQNAgent(state_size=actual_state_size, action_size=action_size, device='cpu')
        agent.load_model(model_path)
        agent.set_eval_mode()
        print(f"Loaded trained model from {model_path}")

    # Telemetry tracking
    power_telemetry = []
    data_telemetry = []
    total_reward = 0
    steps = int(env_config['observation_horizon'] / env_config['time_step'])
    
    for step in range(steps):
        actions = []
        
        if agent_type == "random":
            # Random baseline: Pick random actions
            actions = np.random.randint(0, env_config['max_targets'] + 1, size=env_config['num_satellites'])
            
        elif agent_type == "greedy":
            # Greedy baseline: Always try to look at highest priority active target
            if env.targets:
                # Find highest priority target
                best_target_idx = np.argmax([t.priority * t.value for t in env.targets]) + 1
                actions = [best_target_idx] * env_config['num_satellites']
            else:
                actions = [0] * env_config['num_satellites']
                
        elif agent_type == "dqn":
            # Smart agent: Use the trained neural network
            actions = agent.act_multi_satellite(state, env_config['num_satellites'], training=False)
            
        # Step the environment
        state, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        
        # Collect telemetry
        avg_power = np.mean([sat.power_level for sat in env.satellites])
        total_data = np.sum([sat.data_storage for sat in env.satellites])
        power_telemetry.append(float(avg_power))
        data_telemetry.append(float(total_data))
        
        if terminated or truncated:
            break
            
    stats = info['episode_stats']
    print(f"✅ Simulation Complete | Targets Completed: {stats['targets_completed']} | Reward: {total_reward:.2f}")
    
    return {
        "requests_completed": stats['targets_completed'],
        "total_value_gained": stats['total_value_gained'],
        "total_reward": float(total_reward),
        "avg_power_usage": float(np.mean(power_telemetry)),
        "min_power_drop": float(np.min(power_telemetry)),
        "total_data_collected": float(np.max(data_telemetry)),
        "emergency_situations": stats['emergency_situations'],
        "power_telemetry": power_telemetry,
        "data_telemetry": data_telemetry
    }

def main():
    config = load_config()
    os.makedirs('results', exist_ok=True)
    
    # 1. Find the trained model
    latest_model = find_latest_model()
    if not latest_model:
        print("⚠️ No trained DQN model found! Run `python src/training/train_dqn.py` first.")
        return
        
    print(f"Preparing to benchmark using: {latest_model}")
    
    # 2. Run benchmarks with the SAME SEED for fairness
    benchmark_seed = 2026
    
    results = {}
    results['random'] = run_simulation(config['environment'], agent_type="random", seed=benchmark_seed)
    results['greedy'] = run_simulation(config['environment'], agent_type="greedy", seed=benchmark_seed)
    results['dqn'] = run_simulation(config['environment'], agent_type="dqn", model_path=latest_model, seed=benchmark_seed)
    
    # 3. Save JSON Metrics
    metrics_path = 'results/simulation_metrics.json'
    
    # Create a simplified version to save (excluding massive telemetry arrays to keep JSON clean)
    clean_results = {}
    for agent_name, data in results.items():
        clean_results[agent_name] = {k: v for k, v in data.items() if 'telemetry' not in k}
        
    with open(metrics_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
        
    print(f"\n💾 Saved comparative metrics to {metrics_path}")
    
    # 4. Generate Comparative Bar Chart
    print("📈 Generating Benchmark Visualization...")
    labels = ['Random', 'Greedy', 'Trained DQN']
    completions = [results['random']['requests_completed'], 
                  results['greedy']['requests_completed'], 
                  results['dqn']['requests_completed']]
                  
    rewards = [results['random']['total_reward'], 
              results['greedy']['total_reward'], 
              results['dqn']['total_reward']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Completions
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    bars1 = ax1.bar(labels, completions, color=colors)
    ax1.set_title('Targets Successfully Observed', fontsize=14)
    ax1.set_ylabel('Completed Targets')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add numbers on bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom', fontweight='bold')
        
    # Plot Total Reward
    bars2 = ax2.bar(labels, rewards, color=colors)
    ax2.set_title('Total Simulation Reward (Efficiency + Value)', fontsize=14)
    ax2.set_ylabel('Reward Score')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + (abs(yval)*0.05), int(yval), ha='center', va='bottom', fontweight='bold')
        
    plt.suptitle('AgriSight Satellite Scheduling Benchmark', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = 'results/plots/benchmark_comparison.png'
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved benchmark plot to {plot_path}")
    
    print("\n🏆 Benchmark Complete!")

if __name__ == "__main__":
    main()