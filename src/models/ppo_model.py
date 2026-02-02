"""
PPO (Proximal Policy Optimization) implementation.
Better for continuous games and more stable than DQN.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Any

from src.core import BaseModel


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO"""

    def __init__(self, state_shape: tuple, num_actions: int, hidden_size: int = 512):
        """
        Initialize Actor-Critic network.

        Args:
            state_shape: Shape of input state (e.g., (84, 84, 4))
            num_actions: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super(ActorCriticNetwork, self).__init__()

        # Shared convolutional layers for feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate conv output size
        conv_out_size = self._get_conv_out(state_shape)

        # Shared fully connected layer
        self.shared_fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, num_actions)

        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)

    def _get_conv_out(self, shape):
        """Calculate output size of convolutional layers"""
        o = self.conv(torch.zeros(1, shape[2], shape[0], shape[1]))
        return int(np.prod(o.size()))

    def forward(self, x):
        """Forward pass returning both policy and value"""
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view for non-contiguous tensors
        x = self.shared_fc(x)

        # Policy logits
        policy_logits = self.actor(x)

        # Value estimate
        value = self.critic(x)

        return policy_logits, value

    def get_action_and_value(self, x, action=None):
        """
        Get action, log probability, entropy, and value.

        Args:
            x: State input
            action: Specific action to evaluate (optional)

        Returns:
            action, log_prob, entropy, value
        """
        policy_logits, value = self.forward(x)
        dist = Categorical(logits=policy_logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


class PPOModel(BaseModel):
    """PPO model implementation"""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize PPO model.

        Config should contain:
            - state_shape: tuple (height, width, channels)
            - num_actions: int
            - learning_rate: float (default: 3e-4)
            - gamma: float (default: 0.99)
            - gae_lambda: float (default: 0.95) - GAE parameter
            - clip_epsilon: float (default: 0.2) - PPO clipping parameter
            - value_coef: float (default: 0.5) - Value loss coefficient
            - entropy_coef: float (default: 0.01) - Entropy bonus coefficient
            - max_grad_norm: float (default: 0.5) - Gradient clipping
            - ppo_epochs: int (default: 4) - Number of PPO update epochs
            - minibatch_size: int (default: 64) - Minibatch size for updates
        """
        super().__init__(config)

        self.state_shape = config['state_shape']
        self.num_actions = config['num_actions']

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.minibatch_size = config.get('minibatch_size', 64)
        self.learning_rate = config.get('learning_rate', 3e-4)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Network
        hidden_size = config.get('hidden_size', 512)
        self.network = ActorCriticNetwork(
            self.state_shape, self.num_actions, hidden_size
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # Training state
        self.steps = 0

    def predict(self, state: np.ndarray) -> int:
        """Predict best action for given state"""
        with torch.no_grad():
            state_tensor = self._preprocess_state(state)
            policy_logits, _ = self.network(state_tensor)
            dist = Categorical(logits=policy_logits)
            action = dist.probs.argmax()
            return action.item()

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> tuple[int, float, float]:
        """
        Get action with value and log probability (for training).

        Args:
            state: Current state
            epsilon: Ignored for PPO (kept for compatibility)

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = self._preprocess_state(state)
            action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
            return action.item(), log_prob.item(), value.item()

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """
        Perform PPO update.

        Batch should contain:
            - states: (batch_size, *state_shape)
            - actions: (batch_size,)
            - old_log_probs: (batch_size,)
            - returns: (batch_size,)
            - advantages: (batch_size,)
            - values: (batch_size,) - old value estimates
        """
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
        returns = torch.FloatTensor(batch['returns']).to(self.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Transpose from (B, H, W, C) to (B, C, H, W) and ensure contiguous
        if len(states.shape) == 4:
            states = states.permute(0, 3, 1, 2).contiguous()

        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0

        batch_size = states.shape[0]
        indices = np.arange(batch_size)

        for epoch in range(self.ppo_epochs):
            # Shuffle data
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                # Forward pass
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    mb_states, mb_actions
                )

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                values = values.squeeze()
                value_loss = ((values - mb_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    total_approx_kl += approx_kl.item()

        self.steps += 1

        num_updates = self.ppo_epochs * (batch_size // self.minibatch_size)

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'approx_kl': total_approx_kl / num_updates,
        }

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to tensor with proper shape"""
        # Ensure state is contiguous in memory
        state = np.ascontiguousarray(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # Transpose from (B, H, W, C) to (B, C, H, W) and ensure contiguous
        if len(state_tensor.shape) == 4:
            state_tensor = state_tensor.permute(0, 3, 1, 2).contiguous()
        return state_tensor

    def save(self, path: str) -> None:
        """Save model weights"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)

    def load(self, path: str) -> None:
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint.get('steps', 0)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Array of rewards
        values: Array of value estimates
        next_values: Array of next value estimates
        dones: Array of done flags
        gamma: Discount factor
        gae_lambda: GAE parameter

    Returns:
        (advantages, returns)
    """
    advantages = np.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = next_values[t]
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values

    return advantages, returns
