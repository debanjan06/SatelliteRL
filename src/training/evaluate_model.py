#!/usr/bin/env python3
import os
import sys
import yaml
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.satellite_env import SatelliteEnvironment
from agents.dqn_agent import DQNAgent

def load_config(config_path="configs/default.yml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("🔬 Evaluating Trained AgriSight Model (Episode 3000)")
    config = load_config()
    env_config = config['environment']
    
    env = SatelliteEnvironment(
        num_satellites=env_config['num_satellites'],
        max_targets=env_config['max_targets'],
        observation_horizon=env_config['observation_horizon'],
        time_step=env_config['time_step']
    )
    
    dummy_obs, _ = env.reset()
    state_size = len(dummy_obs)
    action_size = env_config['max_targets'] + 1
    
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Load the specific checkpoint
    model_path = "models/checkpoints/dqn_episode_3000.pth"
    agent.load_model(model_path)
    agent.set_eval_mode()
    
    eval_rewards = []
    eval_completions = []
    
    for episode in range(10):  # Test over 10 full days of simulation
        state, info = env.reset(seed=2000 + episode)
        total_reward = 0
        
        for step in range(config['training']['max_steps_per_episode']):
            actions = agent.act_multi_satellite(state, env_config['num_satellites'], training=False)
            state, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward
            if terminated or truncated:
                break
                
        eval_rewards.append(total_reward)
        eval_completions.append(info['episode_stats']['targets_completed'])
        print(f"Test Day {episode + 1}: Reward = {total_reward:.2f} | Targets Completed = {info['episode_stats']['targets_completed']}")

    print("\n📊 Final Evaluation Results:")
    print(f"Average Reward: {np.mean(eval_rewards):.2f}")
    print(f"Average Completions per Day: {np.mean(eval_completions):.1f}")

if __name__ == "__main__":
    main()