#!/usr/bin/env python3
"""
Deep Q-Network (DQN) Agent for Satellite Constellation Management

This module implements a DQN agent capable of learning optimal scheduling
policies for satellite constellation operations.

Author: Debanjan Shil
Date: June 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import pickle
import os

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Deep Q-Network architecture for satellite scheduling"""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [512, 256, 128]):
        """
        Initialize the DQN network
        
        Args:
            state_size: Size of the input state vector
            action_size: Number of possible actions
            hidden_layers: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build the network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through the network"""
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int, batch_size: int):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            batch_size: Size of batches to sample
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self) -> List[Experience]:
        """Sample a batch of experiences"""
        return random.sample(self.buffer, self.batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
    def can_sample(self) -> bool:
        """Check if buffer has enough experiences for sampling"""
        return len(self.buffer) >= self.batch_size

class DQNAgent:
    """Deep Q-Network agent for satellite constellation management"""
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update: int = 100,
                 device: str = 'auto'):
        """
        Initialize DQN agent
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update: Steps between target network updates
            device: Computing device ('auto', 'cpu', 'cuda')
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 DQN Agent initialized on device: {self.device}")
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        
        # Training metrics
        self.training_step = 0
        self.episode_count = 0
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current environment state
            training: Whether agent is in training mode
            
        Returns:
            Selected action
        """
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            action = random.randrange(self.action_size)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        
        return action
    
    def act_multi_satellite(self, state: np.ndarray, num_satellites: int, training: bool = True) -> List[int]:
        """
        Choose actions for multiple satellites simultaneously
        
        Args:
            state: Current environment state
            num_satellites: Number of satellites in constellation
            training: Whether agent is in training mode
            
        Returns:
            List of actions for each satellite
        """
        actions = []
        
        # For now, use independent action selection for each satellite
        # TODO: Implement coordinated multi-agent action selection
        for sat_idx in range(num_satellites):
            # Extract satellite-specific state features
            # This is a simplified approach - could be improved with attention mechanisms
            action = self.act(state, training)
            actions.append(action)
        
        return actions
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def replay(self) -> Optional[float]:
        """
        Train the agent on a batch of experiences
        
        Returns:
            Training loss if training occurred, None otherwise
        """
        
        if not self.replay_buffer.can_sample():
            return None
        
        # Sample batch of experiences
        experiences = self.replay_buffer.sample()
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
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
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update': self.target_update
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.loss_history = checkpoint['loss_history']
        self.reward_history = checkpoint['reward_history']
        
        print(f"📂 Model loaded from {filepath}")
        print(f"   Training step: {self.training_step}")
        print(f"   Episode count: {self.episode_count}")
        print(f"   Current epsilon: {self.epsilon:.4f}")
    
    def get_training_stats(self) -> dict:
        """Get training statistics"""
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'avg_reward': np.mean(self.reward_history[-100:]) if self.reward_history else 0
        }
    
    def set_eval_mode(self):
        """Set agent to evaluation mode (no exploration)"""
        self.q_network.eval()
        self.epsilon = 0.0
    
    def set_train_mode(self):
        """Set agent to training mode"""
        self.q_network.train()

# Test function for the DQN agent
def test_dqn_agent():
    """Test the DQN agent with dummy data"""
    print("🧪 Testing DQN Agent...")
    
    # Create agent
    state_size = 100  # Example state size
    action_size = 21  # Example action size (0-20 targets)
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=1e-3,
        buffer_size=1000,
        batch_size=32
    )
    
    # Test action selection
    dummy_state = np.random.randn(state_size)
    action = agent.act(dummy_state)
    print(f"Selected action: {action}")
    
    # Test multi-satellite action selection
    multi_actions = agent.act_multi_satellite(dummy_state, num_satellites=3)
    print(f"Multi-satellite actions: {multi_actions}")
    
    # Test experience storage and replay
    for i in range(100):
        state = np.random.randn(state_size)
        action = random.randint(0, action_size - 1)
        reward = random.uniform(-10, 10)
        next_state = np.random.randn(state_size)
        done = random.choice([True, False])
        
        agent.remember(state, action, reward, next_state, done)
    
    # Test training
    loss = agent.replay()
    if loss is not None:
        print(f"Training loss: {loss:.4f}")
    
    # Test model saving/loading
    test_model_path = "models/test_dqn.pth"
    agent.save_model(test_model_path)
    
    # Create new agent and load model
    agent2 = DQNAgent(state_size=state_size, action_size=action_size)
    agent2.load_model(test_model_path)
    
    # Test that loaded agent produces same action
    action2 = agent2.act(dummy_state, training=False)
    print(f"Loaded agent action: {action2}")
    
    # Cleanup
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    
    print("✅ DQN Agent tests completed successfully!")

if __name__ == "__main__":
    test_dqn_agent()