"""
GameInfo dataclass - structured container for game state information.
"""
import numpy as np
from typing import Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from src.utils import GameMemoryMonitor


@dataclass
class GameInfo:
    """
    Structured container for game state information passed to all callback functions.

    This class provides a standardized way to pass game information between
    the environment and user-defined callback functions (extract_info_fn,
    reward_fn, update_state_fn).

    Attributes:
        screenshot: Original raw screenshot captured from the game window
        processed: Preprocessed screenshot/observation (after preprocess_fn)
        memory_monitor: Reference to GameMemoryMonitor for reading game memory
        data: User-defined dictionary for storing custom game state data
              (e.g., health, position, score, enemies, etc.)

    Example usage in callbacks:
        ```python
        def extract_info_fn(game_info: GameInfo) -> GameInfo:
            # Read game memory values
            if game_info.memory_monitor:
                game_info.data['health'] = game_info.memory_monitor.read('health')
                game_info.data['position_x'] = game_info.memory_monitor.read('pos_x')
                game_info.data['score'] = game_info.memory_monitor.read('score')

            # Or extract from screenshot using CV
            game_info.data['enemies_visible'] = detect_enemies(game_info.screenshot)
            return game_info

        def reward_fn(prev_info: GameInfo, curr_info: GameInfo) -> float:
            reward = 0.0
            # Reward for gaining score
            reward += (curr_info.data.get('score', 0) - prev_info.data.get('score', 0)) * 0.1
            # Penalty for losing health
            reward += (curr_info.data.get('health', 0) - prev_info.data.get('health', 0)) * 0.5
            return reward

        def update_state_fn(state: dict, info: GameInfo, action: Any) -> dict:
            state['last_health'] = info.data.get('health', 100)
            state['total_score'] = info.data.get('score', 0)
            return state
        ```
    """
    # Flag to indicate if this is an empty/initial GameInfo instance
    is_empty: bool = False

    # Original raw screenshot from game window
    screenshot: Optional[np.ndarray] = None

    # Preprocessed observation (resized, grayscale, normalized, etc.)
    processed: Optional[np.ndarray] = None

    # Memory monitor for reading game memory (can be None if not used)
    memory_monitor: Optional['GameMemoryMonitor'] = None

    # User-defined data dictionary for custom game state
    # Examples: health, mana, position, score, level, enemies, items, etc.
    data: dict = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Convenience method to get value from data dict."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Convenience method to set value in data dict."""
        self.data[key] = value

    def update(self, values: dict) -> None:
        """Convenience method to update multiple values in data dict."""
        self.data.update(values)

    def copy(self) -> 'GameInfo':
        """Create a shallow copy of GameInfo with a new data dict."""
        return GameInfo(
            screenshot=self.screenshot,
            processed=self.processed,
            memory_monitor=self.memory_monitor,
            data=self.data.copy()
        )

    @property
    def game_over(self) -> bool:
        """Check if game_over flag is set in data."""
        return self.data.get('game_over', False)

    @property
    def mission_complete(self) -> bool:
        """Check if mission_complete flag is set in data."""
        return self.data.get('mission_complete', False)
