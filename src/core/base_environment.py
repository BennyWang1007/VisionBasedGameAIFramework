"""
Base abstract class for game environment.
This wraps the actual game and provides RL interface.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Callable  # , Optional

from .game_info import GameInfo


class BaseEnvironment(ABC):
    """Abstract base class for game environment"""

    def __init__(
        self,
        preprocess_fn: Callable[[np.ndarray], np.ndarray],
        extract_info_fn: Callable[[GameInfo], GameInfo],
        reward_fn: Callable[[GameInfo, GameInfo], float],
        update_state_fn: Callable[[dict, GameInfo, Any], dict]
    ):
        """
        Initialize environment with user-defined callbacks.

        Args:
            preprocess_fn: Function to preprocess raw screenshots
            extract_info_fn: Function to extract game information, receives GameInfo
            reward_fn: Function to calculate reward from previous and current GameInfo
            update_state_fn: Function to update state based on GameInfo
        """
        self.preprocess_fn = preprocess_fn
        self.extract_info_fn = extract_info_fn
        self.reward_fn = reward_fn
        self.update_state_fn = update_state_fn

        self.current_state: dict = {}
        self.previous_info: GameInfo = GameInfo(is_empty=True)

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, GameInfo]:
        """
        Execute action and return results.

        Args:
            action: Action to execute

        Returns:
            observation: Next state observation
            reward: Reward received
            done: Whether episode is finished
            info: GameInfo with game state information
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

    def _get_screenshot(self) -> np.ndarray:
        """
        Capture screenshot from game window.
        To be implemented by concrete class.

        Returns:
            Raw screenshot as numpy array
        """
        raise NotImplementedError

    def _execute_action(self, action: Any) -> None:
        """
        Execute action in the game (send inputs via Windows API).
        To be implemented by concrete class.

        Args:
            action: Action to execute
        """
        raise NotImplementedError
