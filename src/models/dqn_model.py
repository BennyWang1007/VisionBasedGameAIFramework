"""
Example DQN model implementation.
This is a placeholder - you should customize based on your needs.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any

from src.core import BaseModel


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""

    def __init__(self, state_shape: tuple, num_actions: int):
        """
        Initialize DQN network.

        Args:
            state_shape: Shape of input state (e.g., (84, 84, 4) for stacked frames)
            num_actions: Number of possible actions
        """
        super(DQNNetwork, self).__init__()

        # Convolutional layers for image input
        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate size after convolutions
        conv_out_size = self._get_conv_out(state_shape)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        """Calculate output size of convolutional layers"""
        o = self.conv(torch.zeros(1, shape[2], shape[0], shape[1]))
        return int(np.prod(o.size()))

    def forward(self, x):
        """Forward pass"""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DQNModel(BaseModel):
    """DQN model implementation"""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize DQN model.

        Config should contain:
            - state_shape: tuple (height, width, channels)
            - num_actions: int
            - learning_rate: float
            - gamma: float (discount factor)
        """
        super().__init__(config)

        self.state_shape = config['state_shape']
        self.num_actions = config['num_actions']
        self.gamma = config.get('gamma', 0.99)
        self.learning_rate = config.get('learning_rate', 0.00025)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks
        self.policy_net = DQNNetwork(self.state_shape, self.num_actions).to(self.device)
        self.target_net = DQNNetwork(self.state_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Training parameters
        self.update_target_every = config.get('update_target_every', 1000)
        self.steps = 0

    def predict(self, state: np.ndarray) -> int:
        """Predict best action for given state"""
        with torch.no_grad():
            state_tensor = self._preprocess_state(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Get action with epsilon-greedy exploration"""
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        return self.predict(state)

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Perform one training step"""
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)

        # Ensure proper shape for conv layers
        if len(states.shape) == 3:
            # Add batch dimension if needed
            states = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)

        # Transpose from (B, H, W, C) to (B, C, H, W)
        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            'loss': loss.item(),
            'mean_q_value': current_q_values.mean().item()
        }

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to tensor with proper shape"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # Transpose from (B, H, W, C) to (B, C, H, W)
        if len(state_tensor.shape) == 4:
            state_tensor = state_tensor.permute(0, 3, 1, 2)
        return state_tensor

    def save(self, path: str) -> None:
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)

    def load(self, path: str) -> None:
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint.get('steps', 0)
