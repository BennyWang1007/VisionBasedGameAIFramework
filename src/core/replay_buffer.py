"""
Experience replay buffer for storing and sampling transitions.
"""
import numpy as np
from collections import deque
import random
from typing import Any


class ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms"""

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer: deque[tuple[Any, ...]] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing batched transitions
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones)
        }

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
