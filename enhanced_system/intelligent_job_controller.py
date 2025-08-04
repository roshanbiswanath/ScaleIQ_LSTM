"""
Intelligent Job Allocation Controller using Deep Reinforcement Learning
======================================================================

This module implements a sophisticated auto-scaling controller that uses Deep RL (PPO)
to make optimal job allocation decisions based on event forecasts and system state.

Key Features:
- Multi-objective optimization (latency, cost, SLA compliance)
- Real-time adaptation to changing conditions
- Graceful scaling with proper constraints
- Learning from experience and feedback
- Risk-aware decision making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ControllerConfig:
    """Configuration for the job allocation controller."""
    # Environment configuration
    max_jobs: int = 100
    min_jobs: int = 1
    max_job_change: int = 10  # Maximum change per decision
    decision_interval: int = 6  # Every 12 minutes (6 * 2min intervals)
    
    # State space configuration
    forecast_horizons: List[int] = field(default_factory=lambda: [6, 12, 24, 48])
    history_length: int = 24  # 48 minutes of history
    
    # Reward configuration
    sla_threshold: float = 5.0  # seconds
    cost_per_job: float = 1.0
    sla_penalty: float = 100.0
    efficiency_bonus: float = 10.0
    
    # PPO configuration
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Network architecture
    hidden_dim: int = 256
    n_layers: int = 3


class EventProcessingEnvironment(gym.Env):
    """
    Custom environment for event processing auto-scaling.
    
    State: Current system metrics + forecasted demand + historical performance
    Action: Job count adjustment (-max_change to +max_change)
    Reward: Multi-objective function balancing performance, cost, and SLA
    """
    
    def __init__(self, config: ControllerConfig, forecaster=None):
        super().__init__()
        self.config = config
        self.forecaster = forecaster
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-config.max_job_change, 
            high=config.max_job_change, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # State space: [current_metrics, forecasts, history]
        state_dim = (
            8 +  # Current system metrics
            len(config.forecast_horizons) * 4 +  # Forecasts with uncertainty
            config.history_length * 3 +  # Historical performance
            2  # Time features
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=np.float32
        )
        
        # Environment state
        self.current_jobs = config.min_jobs
        self.time_step = 0
        self.performance_history = deque(maxlen=config.history_length)
        self.cost_history = deque(maxlen=config.history_length)
        self.sla_history = deque(maxlen=config.history_length)
        
        # Simulation parameters (would be replaced with real system metrics)
        self.processing_capacity_per_job = 100  # events per interval per job
        self.base_latency = 1.0  # seconds
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_jobs = self.config.min_jobs
        self.time_step = 0
        self.performance_history.clear()
        self.cost_history.clear()
        self.sla_history.clear()
        
        # Initialize with neutral values
        for _ in range(self.config.history_length):
            self.performance_history.append(0.5)
            self.cost_history.append(self.current_jobs * self.config.cost_per_job)
            self.sla_history.append(1.0)  # SLA compliance
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        # Parse action (job count change)
        job_change = np.clip(action[0], -self.config.max_job_change, self.config.max_job_change)
        job_change = int(np.round(job_change))
        
        # Update job count with constraints
        new_jobs = np.clip(
            self.current_jobs + job_change,
            self.config.min_jobs,
            self.config.max_jobs
        )
        
        # Simulate system performance (in real system, this would be measured)
        performance_metrics = self._simulate_performance(new_jobs)
        
        # Calculate reward
        reward = self._calculate_reward(new_jobs, performance_metrics, job_change)
        
        # Update state
        self.current_jobs = new_jobs
        self.time_step += 1
        
        # Store metrics in history
        self.performance_history.append(performance_metrics['efficiency'])
        self.cost_history.append(new_jobs * self.config.cost_per_job)
        self.sla_history.append(performance_metrics['sla_compliance'])
        
        # Episode termination (for training purposes)
        done = self.time_step >= 1000  # Long episodes for continuous learning
        
        info = {
            'jobs': new_jobs,
            'efficiency': performance_metrics['efficiency'],
            'latency': performance_metrics['latency'],
            'sla_compliance': performance_metrics['sla_compliance'],
            'cost': new_jobs * self.config.cost_per_job,
            'job_change': job_change
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Current system metrics
        current_metrics = [
            self.current_jobs / self.config.max_jobs,  # Normalized job count
            len(self.performance_history) / self.config.history_length,  # History fill ratio
            np.mean(self.performance_history) if self.performance_history else 0.5,
            np.std(self.performance_history) if len(self.performance_history) > 1 else 0,
            np.mean(self.sla_history) if self.sla_history else 1.0,
            np.mean(self.cost_history) if self.cost_history else 0,
            (self.current_jobs - self.config.min_jobs) / (self.config.max_jobs - self.config.min_jobs),
            min(self.time_step / 100, 1.0)  # Normalized time
        ]
        
        # Forecasted demand (would come from forecaster in real system)
        if self.forecaster:
            # Get forecasts from model
            forecasts = [0.5] * len(self.config.forecast_horizons) * 4  # Placeholder
        else:
            # Simulated forecasts
            forecasts = [
                0.5 + 0.3 * np.sin(self.time_step * 0.1),  # Demand
                0.2,  # Uncertainty
                0.1,  # Trend
                0.0   # Anomaly score
            ] * len(self.config.forecast_horizons)
        
        # Historical performance features
        history_features = (
            list(self.performance_history) +
            list(self.cost_history) +
            list(self.sla_history)
        )
        
        # Pad if necessary
        while len(history_features) < self.config.history_length * 3:
            history_features.append(0.0)
        
        # Time features
        time_features = [
            np.sin(2 * np.pi * (self.time_step % 720) / 720),  # Daily cycle
            np.cos(2 * np.pi * (self.time_step % 720) / 720)   # Daily cycle
        ]
        
        # Combine all features
        state = np.array(current_metrics + forecasts + history_features + time_features, dtype=np.float32)
        
        return state
    
    def _simulate_performance(self, jobs: int) -> Dict[str, float]:
        """Simulate system performance given job count."""
        # Simulate current demand (in real system, this would be observed)
        base_demand = 1000 + 500 * np.sin(self.time_step * 0.1)  # Cyclical demand
        demand = max(0, base_demand + np.random.normal(0, 100))  # Add noise
        
        # Calculate processing capacity
        total_capacity = jobs * self.processing_capacity_per_job
        
        # Calculate utilization and performance
        utilization = min(demand / max(total_capacity, 1), 1.0)
        
        # Latency increases with utilization (queueing theory)
        if utilization < 0.8:
            latency = self.base_latency * (1 + utilization)
        else:
            # Non-linear increase at high utilization
            latency = self.base_latency * (1 + utilization + (utilization - 0.8) ** 3 * 10)
        
        # Efficiency (processed events per job)
        efficiency = min(demand / max(jobs, 1), self.processing_capacity_per_job) / self.processing_capacity_per_job
        
        # SLA compliance (percentage of requests under threshold)
        sla_compliance = 1.0 if latency <= self.config.sla_threshold else max(0, 1 - (latency - self.config.sla_threshold) / self.config.sla_threshold)
        
        return {
            'demand': demand,
            'utilization': utilization,
            'latency': latency,
            'efficiency': efficiency,
            'sla_compliance': sla_compliance
        }
    
    def _calculate_reward(self, jobs: int, performance: Dict[str, float], job_change: int) -> float:
        """Calculate multi-objective reward."""
        # SLA compliance reward/penalty
        sla_reward = performance['sla_compliance'] * 10
        if performance['sla_compliance'] < 0.95:
            sla_reward -= self.config.sla_penalty * (0.95 - performance['sla_compliance'])
        
        # Cost penalty (linear with job count)
        cost_penalty = jobs * self.config.cost_per_job * 0.1
        
        # Efficiency bonus
        efficiency_reward = performance['efficiency'] * self.config.efficiency_bonus
        
        # Stability bonus (penalty for frequent changes)
        stability_bonus = -abs(job_change) * 0.5
        
        # Utilization reward (encourage optimal utilization around 70-80%)
        target_utilization = 0.75
        utilization_reward = -abs(performance['utilization'] - target_utilization) * 5
        
        # Total reward
        total_reward = (
            sla_reward +
            efficiency_reward +
            utilization_reward +
            stability_bonus -
            cost_penalty
        )
        
        return total_reward


class PPONetwork(nn.Module):
    """PPO Actor-Critic network for job allocation decisions."""
    
    def __init__(self, state_dim: int, action_dim: int, config: ControllerConfig):
        super().__init__()
        self.config = config
        
        # Shared feature extractor
        self.shared_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for _ in range(config.n_layers):
            layer = nn.Sequential(
                nn.Linear(prev_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.shared_layers.append(layer)
            prev_dim = config.hidden_dim
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, action_dim * 2)  # mean and std
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through actor-critic network."""
        # Shared feature extraction
        x = state
        for layer in self.shared_layers:
            x = layer(x)
        
        # Actor output (policy parameters)
        actor_out = self.actor(x)
        mean = torch.tanh(actor_out[..., :1]) * self.config.max_job_change  # Constrain to action space
        std = F.softplus(actor_out[..., 1:]) + 1e-8  # Ensure positive std
        
        # Critic output (value)
        value = self.critic(x)
        
        return mean, std, value
    
    def get_action_and_value(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action and value for given state."""
        mean, std, value = self.forward(state)
        
        # Create action distribution
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value, dist.entropy()


class IntelligentJobController:
    """
    Main controller class that orchestrates the reinforcement learning agent
    for intelligent job allocation decisions.
    """
    
    def __init__(self, config: ControllerConfig, forecaster=None):
        self.config = config
        self.forecaster = forecaster
        
        # Create environment
        self.env = EventProcessingEnvironment(config, forecaster)
        
        # Create PPO network
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.network = PPONetwork(state_dim, action_dim, config)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
        # Training storage
        self.reset_storage()
        
    def reset_storage(self):
        """Reset storage for training data."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[int, Dict]:
        """Select action given current state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value, entropy = self.network.get_action_and_value(state_tensor)
        
        action_np = action.squeeze().numpy()
        
        if training:
            # Store for training
            self.states.append(state)
            self.actions.append(action_np)
            self.log_probs.append(log_prob.squeeze().numpy())
            self.values.append(value.squeeze().numpy())
        
        info = {
            'value_estimate': value.item(),
            'entropy': entropy.item(),
            'action_mean': action_np,
            'raw_action': action_np
        }
        
        return int(np.round(np.clip(action_np, -self.config.max_job_change, self.config.max_job_change))), info
    
    def train_episode(self, max_steps: int = 1000) -> Dict[str, float]:
        """Train for one episode."""
        state = self.env.reset()
        self.reset_storage()
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            action, _ = self.select_action(state, training=True)
            
            # Execute action
            next_state, reward, done, info = self.env.step([action])
            
            # Store experience
            self.rewards.append(reward)
            self.dones.append(done)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if done:
                break
        
        # Train on collected experience
        training_metrics = self._train_on_batch()
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            **training_metrics
        }
    
    def _train_on_batch(self) -> Dict[str, float]:
        """Train PPO on collected batch of experience."""
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        values = torch.FloatTensor(np.array(self.values))
        rewards = torch.FloatTensor(np.array(self.rewards))
        dones = torch.FloatTensor(np.array(self.dones))
        
        # Calculate advantages using GAE
        advantages = self._calculate_gae(rewards, values, dones)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        n_updates = 4  # Number of PPO epochs
        
        for _ in range(n_updates):
            # Get current policy
            _, new_log_probs, new_values, entropy = self.network.get_action_and_value(states)
            new_log_probs = new_log_probs.squeeze()
            new_values = new_values.squeeze()
            
            # Calculate ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Policy loss (clipped)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # Entropy loss (for exploration)
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = (
                policy_loss + 
                self.config.value_coef * value_loss + 
                self.config.entropy_coef * entropy_loss
            )
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item()
        }
    
    def _calculate_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage
        
        return advantages
    
    def decide_job_allocation(self, current_state: Dict) -> Tuple[int, Dict]:
        """
        Main decision function for production use.
        
        Args:
            current_state: Dictionary containing current system metrics and forecasts
            
        Returns:
            Tuple of (job_count_change, decision_info)
        """
        # Convert current state to model input
        state_vector = self._state_dict_to_vector(current_state)
        
        # Get action from trained model
        action, info = self.select_action(state_vector, training=False)
        
        return action, info
    
    def _state_dict_to_vector(self, state_dict: Dict) -> np.ndarray:
        """Convert state dictionary to vector for model input."""
        # This would need to be implemented based on the specific state format
        # For now, return a placeholder
        return np.zeros(self.env.observation_space.shape[0])


if __name__ == "__main__":
    # Example usage
    config = ControllerConfig()
    controller = IntelligentJobController(config)
    
    print("Intelligent Job Allocation Controller")
    print(f"State dimension: {controller.env.observation_space.shape[0]}")
    print(f"Action dimension: {controller.env.action_space.shape[0]}")
    print(f"Network parameters: {sum(p.numel() for p in controller.network.parameters()):,}")
    print("✅ Controller ready for training!")
