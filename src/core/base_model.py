"""
Base abstract class for RL models.
Users should inherit from this and implement their specific model architecture.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class BaseModel(ABC):
    """Abstract base class for RL models (e.g., DQN, PPO, A2C, etc.)"""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the model.

        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config

    @abstractmethod
    def predict(self, state: np.ndarray) -> Any:
        """
        Predict action(s) given a state.

        Args:
            state: Current state observation

        Returns:
            Action(s) to take
        """
        pass

    @abstractmethod
    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Dictionary containing 'states', 'actions', 'rewards', 'next_states', 'dones'

        Returns:
            Dictionary of training metrics (e.g., loss, accuracy)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights."""
        pass

    @abstractmethod
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> Any:
        """
        Get action with exploration (e.g., epsilon-greedy).

        Args:
            state: Current state
            epsilon: Exploration rate

        Returns:
            Selected action
        """
        pass
