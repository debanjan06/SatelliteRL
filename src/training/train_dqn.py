#!/usr/bin/env python3
"""
Fixed Training Script for SatelliteRL DQN Agent
Handles observation space size mismatch and tensor conversion warnings.

Author: Debanjan Shil
Date: June 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import json
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.satellite_env import SatelliteEnvironment
from agents.dqn_agent import DQNAgent

def load_config(config_path="configs/default.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        # Return default config
        return {
            'environment': {
                'num_satellites': 3,
                'max_targets': 10,
                'observation_horizon': 24.0,
                'time_step': 0.1
            },
            'agent': {
                'hyperparameters': {
                    'learning_rate': 0.0001,
                    'gamma': 0.99,
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.01,
                    'epsilon_decay': 0.995
                },
                'replay_buffer': {
                    'capacity': 10000,
                    'batch_size': 32
                },
                'training': {
                    'target_update_frequency': 100
                }
            },
            'training': {
                'max_episodes': 500,  # Reduced for faster testing
                'max_steps_per_episode': 240
            }
        }

class FixedDQNAgent(DQNAgent):
    """Fixed DQN Agent with proper tensor handling"""
    
    def replay(self):
        """Fixed replay method with proper tensor conversion"""
        
        if not self.replay_buffer.can_sample():
            return None
        
        # Sample batch of experiences
        experiences = self.replay_buffer.sample()
        
        # Convert to numpy arrays first, then to tensors (fixes warning)
        states_np = np.array([e.state for e in experiences], dtype=np.float32)
        actions_np = np.array([e.action for e in experiences], dtype=np.int64)
        rewards_np = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states_np = np.array([e.next_state for e in experiences], dtype=np.float32)
        dones_np = np.array([e.done for e in experiences], dtype=bool)
        
        # Convert to tensors
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions_np).to(self.device)
        rewards = torch.FloatTensor(rewards_np).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.BoolTensor(dones_np).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update training metrics
        self.training_step += 1
        self.loss_history.append(loss.item())
        
        # Update target network periodically
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"🎯 Target network updated at step {self.training_step}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.epsilon_history.append(self.epsilon)
        
        return loss.item()

def train_dqn_agent(config):
    """Train DQN agent on satellite constellation environment"""
    
    print("🚀 Starting SatelliteRL Training")
    print("=" * 60)
    
    # Create environment
    env_config = config['environment']
    env = SatelliteEnvironment(
        num_satellites=env_config['num_satellites'],
        max_targets=env_config['max_targets'],
        observation_horizon=env_config['observation_horizon'],
        time_step=env_config['time_step']
    )
    
    # Get actual observation space size
    dummy_obs, _ = env.reset()
    actual_state_size = len(dummy_obs)
    
    print(f"🌍 Environment created:")
    print(f"   Satellites: {env_config['num_satellites']}")
    print(f"   Max targets: {env_config['max_targets']}")
    print(f"   Actual observation size: {actual_state_size}")
    print(f"   Action space: {env.action_space}")
    
    # Create agent with correct state size
    agent_config = config['agent']
    action_size = env_config['max_targets'] + 1  # +1 for no-action
    
    agent = FixedDQNAgent(  # Use fixed agent
        state_size=actual_state_size,  # Use actual size
        action_size=action_size,
        lr=agent_config['hyperparameters']['learning_rate'],
        gamma=agent_config['hyperparameters']['gamma'],
        epsilon=agent_config['hyperparameters']['epsilon_start'],
        epsilon_min=agent_config['hyperparameters']['epsilon_end'],
        epsilon_decay=agent_config['hyperparameters']['epsilon_decay'],
        buffer_size=agent_config['replay_buffer']['capacity'],
        batch_size=agent_config['replay_buffer']['batch_size'],
        target_update=agent_config['training']['target_update_frequency']
    )
    
    print(f"🤖 Agent created:")
    print(f"   State size: {actual_state_size}")
    print(f"   Action size: {action_size}")
    print(f"   Device: {agent.device}")
    
    # Training parameters
    training_config = config['training']
    max_episodes = training_config['max_episodes']
    max_steps = training_config['max_steps_per_episode']
    
    # Training metrics
    episode_rewards = []
    episode_completions = []
    episode_losses = []
    best_reward = float('-inf')
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Max episodes: {max_episodes}")
    print(f"   Max steps per episode: {max_steps}")
    print(f"   Target update frequency: {agent.target_update}")
    
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    for episode in range(max_episodes):
        # Reset environment
        state, info = env.reset(seed=episode)
        total_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Simplified approach: single action for first satellite, others idle
            # This makes training more stable initially
            action = agent.act(state, training=True)
            actions = [action, 0, 0]  # First satellite acts, others idle
            
            # Environment step
            next_state, reward, terminated, truncated, info = env.step(actions)
            
            # Store experience
            agent.remember(state, action, reward, next_state, terminated or truncated)
            
            # Train agent
            if len(agent.replay_buffer) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_loss.append(loss)
            
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        # Update episode count
        agent.episode_count = episode + 1
        agent.reward_history.append(total_reward)
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_completions.append(info['episode_stats']['targets_completed'])
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            os.makedirs('models/best_models', exist_ok=True)
            agent.save_model('models/best_models/best_dqn.pth')
        
        # Periodic logging
        if (episode + 1) % 25 == 0:  # More frequent logging
            avg_reward = np.mean(episode_rewards[-25:])
            avg_completion = np.mean(episode_completions[-25:])
            avg_loss_recent = np.mean([l for l in episode_losses[-25:] if l > 0])
            
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Completed: {episode_completions[-1]:2d} | "
                  f"Avg Completed: {avg_completion:4.1f} | "
                  f"Loss: {avg_loss_recent:6.4f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
        
        # Save checkpoint
        if (episode + 1) % 100 == 0:
            os.makedirs('models/checkpoints', exist_ok=True)
            agent.save_model(f'models/checkpoints/dqn_episode_{episode + 1}.pth')
    
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    
    # Final metrics
    final_avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
    final_avg_completion = np.mean(episode_completions[-50:]) if len(episode_completions) >= 50 else np.mean(episode_completions)
    
    print(f"📊 Final Performance:")
    print(f"   Average Reward (last 50): {final_avg_reward:.2f}")
    print(f"   Average Completions (last 50): {final_avg_completion:.1f}")
    print(f"   Best Episode Reward: {best_reward:.2f}")
    print(f"   Total Training Steps: {agent.training_step}")
    
    return agent, episode_rewards, episode_completions, episode_losses

def plot_training_results(episode_rewards, episode_completions, episode_losses):
    """Plot training results"""
    
    print("\n📈 Generating training plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(episode_rewards, alpha=0.6, color='blue', label='Episode Reward')
    # Moving average
    window = min(25, len(episode_rewards) // 4)
    if len(episode_rewards) >= window and window > 1:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 'red', linewidth=2, label=f'{window}-Episode Average')
    ax1.set_title('Training Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Completion rates
    ax2.plot(episode_completions, alpha=0.6, color='green', label='Targets Completed')
    if len(episode_completions) >= window and window > 1:
        moving_avg_comp = np.convolve(episode_completions, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_completions)), moving_avg_comp, 'darkgreen', linewidth=2, label=f'{window}-Episode Average')
    ax2.set_title('Target Completion Over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Targets Completed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training loss
    valid_losses = [loss for loss in episode_losses if loss > 0]
    if valid_losses:
        ax3.plot(valid_losses, alpha=0.6, color='orange', label='Training Loss')
        if len(valid_losses) >= 10:
            loss_window = min(10, len(valid_losses) // 3)
            moving_avg_loss = np.convolve(valid_losses, np.ones(loss_window)/loss_window, mode='valid')
            ax3.plot(range(loss_window-1, len(valid_losses)), moving_avg_loss, 'red', linewidth=2, label=f'{loss_window}-Episode Average')
    ax3.set_title('Training Loss Over Time')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    if valid_losses:
        ax3.set_yscale('log')
    
    # Performance comparison
    episodes_to_show = min(len(episode_rewards), 200)
    recent_rewards = episode_rewards[-episodes_to_show:]
    recent_completions = episode_completions[-episodes_to_show:]
    
    ax4.scatter(recent_rewards, recent_completions, alpha=0.6, c=range(len(recent_rewards)), 
                cmap='viridis', label='Episode Performance')
    ax4.set_title('Reward vs. Target Completion')
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Targets Completed')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/training_results.png', dpi=300, bbox_inches='tight')
    print("✅ Training plots saved to results/plots/training_results.png")
    
    # Also save individual metrics
    np.save('results/episode_rewards.npy', episode_rewards)
    np.save('results/episode_completions.npy', episode_completions)
    print("✅ Training data saved to results/")
    
    plt.show()

def evaluate_trained_agent(agent, config):
    """Evaluate the trained agent"""
    
    print("\n🔬 Evaluating Trained Agent...")
    
    env_config = config['environment']
    env = SatelliteEnvironment(
        num_satellites=env_config['num_satellites'],
        max_targets=env_config['max_targets'],
        observation_horizon=env_config['observation_horizon'],
        time_step=env_config['time_step']
    )
    
    agent.set_eval_mode()  # No exploration
    
    eval_rewards = []
    eval_completions = []
    
    for episode in range(5):  # 5 evaluation episodes
        state, info = env.reset(seed=1000 + episode)  # Different seed for evaluation
        total_reward = 0
        
        for step in range(config['training']['max_steps_per_episode']):
            action = agent.act(state, training=False)  # No exploration
            actions = [action, 0, 0]  # Consistent with training
            
            state, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        eval_rewards.append(total_reward)
        eval_completions.append(info['episode_stats']['targets_completed'])
        
        print(f"   Eval Episode {episode + 1}: Reward={total_reward:.2f}, Completed={info['episode_stats']['targets_completed']}")
    
    avg_eval_reward = np.mean(eval_rewards)
    avg_eval_completion = np.mean(eval_completions)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Average Reward: {avg_eval_reward:.2f} ± {np.std(eval_rewards):.2f}")
    print(f"   Average Completions: {avg_eval_completion:.1f} ± {np.std(eval_completions):.1f}")
    
    return avg_eval_reward, avg_eval_completion

def main():
    """Main training function"""
    
    # Load configuration
    config = load_config()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        # Train agent
        agent, episode_rewards, episode_completions, episode_losses = train_dqn_agent(config)
        
        # Plot results
        plot_training_results(episode_rewards, episode_completions, episode_losses)
        
        # Evaluate trained agent
        eval_reward, eval_completion = evaluate_trained_agent(agent, config)
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f'models/final_dqn_{timestamp}.pth'
        agent.save_model(final_model_path)
        
        # Save training metrics
        training_results = {
            'config': config,
            'training_episodes': len(episode_rewards),
            'final_epsilon': agent.epsilon,
            'training_steps': agent.training_step,
            'max_reward': max(episode_rewards) if episode_rewards else 0,
            'final_avg_reward': np.mean(episode_rewards[-25:]) if len(episode_rewards) >= 25 else np.mean(episode_rewards),
            'final_avg_completion': np.mean(episode_completions[-25:]) if len(episode_completions) >= 25 else np.mean(episode_completions),
            'evaluation': {
                'avg_reward': eval_reward,
                'avg_completion': eval_completion
            },
            'model_path': final_model_path,
            'timestamp': timestamp
        }
        
        results_path = f'results/training_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        print(f"\n💾 Final model saved to: {final_model_path}")
        print(f"📊 Training results saved to: {results_path}")
        print("\n🎉 Training pipeline completed successfully!")
        
        # Print summary
        print(f"\n🏆 TRAINING SUMMARY:")
        print(f"   Episodes trained: {len(episode_rewards)}")
        print(f"   Best reward: {max(episode_rewards):.2f}")
        print(f"   Final avg reward: {training_results['final_avg_reward']:.2f}")
        print(f"   Final avg completions: {training_results['final_avg_completion']:.1f}")
        print(f"   Training steps: {agent.training_step}")
        
        print("\n🚀 Next steps:")
        print("   1. Analyze the training plots for learning patterns")
        print("   2. Try different hyperparameters for better performance")
        print("   3. Implement multi-satellite coordination")
        print("   4. Add real orbital mechanics integration")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Debugging tips:")
        print("   1. Check if all dependencies are installed")
        print("   2. Verify environment and agent compatibility")
        print("   3. Try reducing batch size or episode count")

if __name__ == "__main__":
    main()