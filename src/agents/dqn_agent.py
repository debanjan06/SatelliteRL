#!/usr/bin/env python3
"""
Deep Q-Network (DQN) Agent for Satellite Constellation Management

This module implements a DQN agent capable of learning optimal scheduling
policies for satellite constellation operations.

Author: Debanjan Shil
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import os

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [512, 256, 128]):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        layers = []
        layers.append(nn.Linear(state_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        layers.append(nn.Linear(hidden_layers[-1], action_size))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        return self.network(state)

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, batch_size: int, alpha: float = 0.6):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, np.array(weights, dtype=np.float32)
        
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = float(prio)
    
    def can_sample(self) -> bool:
        return len(self.buffer) >= self.batch_size
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
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
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"DQN Agent initialized on device: {self.device}")
        
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, batch_size)
        
        self.training_step = 0
        self.episode_count = 0
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if training and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        return action
    
    def act_multi_satellite(self, state: np.ndarray, num_satellites: int, training: bool = True) -> List[int]:
        actions = []
        satellite_state_size = 15
        
        for sat_idx in range(num_satellites):
            masked_state = state.copy()
            for i in range(num_satellites):
                if i != sat_idx:
                    start_idx = i * satellite_state_size
                    end_idx = start_idx + satellite_state_size
                    masked_state[start_idx:end_idx] = 0.0
                    
            action = self.act(masked_state, training)
            actions.append(action)
        return actions
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def replay(self) -> Optional[float]:
        pass
    
    def save_model(self, filepath: str):
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
    
    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # FIX: Added weights_only=False for PyTorch 2.6+ compatibility
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.loss_history = checkpoint['loss_history']
        self.reward_history = checkpoint['reward_history']
    
    def get_training_stats(self) -> dict:
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'avg_reward': np.mean(self.reward_history[-100:]) if self.reward_history else 0
        }
    
    def set_eval_mode(self):
        self.q_network.eval()
        self.epsilon = 0.0
    
    def set_train_mode(self):
        self.q_network.train()