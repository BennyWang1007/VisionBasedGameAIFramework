"""
Concrete implementation of game environment for continuous/running games.
Designed for games that don't need traditional resets - they run continuously.
"""
import numpy as np
from typing import Any, Callable, Optional
import time

from src.core import BaseEnvironment, GameInfo
from src.utils import WindowsGameController, GameMemoryMonitor


class ContinuousGameEnvironment(BaseEnvironment):
    """
    Game environment for continuously running games.
    No traditional reset - game keeps running, we just observe and act.
    """

    def __init__(
        self,
        window_name: str,
        process_name: str,
        preprocess_fn: Callable[[np.ndarray], np.ndarray],
        extract_info_fn: Callable[[GameInfo], GameInfo],
        reward_fn: Callable[[GameInfo, GameInfo], float],
        update_state_fn: Callable[[dict, GameInfo, Any], dict],
        action_space: list,
        use_memory_reading: bool = True,
        frame_skip: int = 4,
        action_delay: float = 0.05,
        episode_timeout: Optional[int] = None
    ):
        """
        Initialize continuous game environment.

        Args:
            window_name: Name of the game window
            process_name: Process name for memory reading (e.g., "game.exe")
            preprocess_fn: Preprocessing function for screenshots
            extract_info_fn: Info extraction - receives GameInfo, should populate data dict
            reward_fn: Reward calculation - receives (previous_info, current_info) GameInfo objects
            update_state_fn: State update - receives (state, info, action)
            action_space: List of possible actions (e.g., ['w', 'a', 's', 'd', 'space'])
            use_memory_reading: Whether to use memory reading (recommended)
            frame_skip: Number of frames to skip between actions
            action_delay: Delay after executing action (seconds)
            episode_timeout: Max steps per episode (None for infinite)
        """
        super().__init__(preprocess_fn, extract_info_fn, reward_fn, update_state_fn)

        # Windows control
        self.controller = WindowsGameController(window_name=window_name)

        # State tracking
        self.mouse_position = (0, 0)

        # Memory reading
        self.use_memory = use_memory_reading
        self.memory_monitor: Optional[GameMemoryMonitor] = None
        if use_memory_reading:
            self.memory_monitor = GameMemoryMonitor(process_name)

        self.action_space = action_space
        self.frame_skip = frame_skip
        self.action_delay = action_delay
        self.episode_timeout = episode_timeout

        # Episode tracking
        self.episode_steps = 0

        # Find game window
        if not self.controller.find_window():
            raise RuntimeError(f"Could not find window: {window_name}")

    def reset(self) -> np.ndarray:
        """
        'Reset' for continuous games - just capture current state.
        The game keeps running; we don't actually reset it.

        For games with time control, you could:
        - Call game API to set timescale to 0
        - Send specific key to pause
        - Wait for certain game state

        Returns:
            Initial observation (preprocessed screenshot)
        """
        # Reset episode counter
        self.episode_steps = 0

        # Optional: Pause game if you have that capability
        # self.controller.send_key('p')  # Example: pause key
        # time.sleep(0.1)

        # Capture current state
        screenshot = self._get_screenshot()
        processed = self.preprocess_fn(screenshot)

        # Create GameInfo and extract info (with memory reading if enabled)
        game_info = GameInfo(
            screenshot=screenshot,
            processed=processed,
            memory_monitor=self.memory_monitor
        )
        game_info = self.extract_info_fn(game_info)

        # Initialize state
        self.current_state = self.update_state_fn({}, game_info, None)
        self.previous_info = game_info

        # Optional: Unpause game
        # self.controller.send_key('p')

        return processed

    # def step(self, action: int) -> tuple[np.ndarray, float, bool, GameInfo]:
    def step(self, action: str | dict | int) -> tuple[np.ndarray, float, bool, GameInfo]:
        """
        Execute action in the continuously running game.

        Args:
            action: Index into action_space or action key/dict

        Returns:
            observation, reward, done, GameInfo
        """

        if isinstance(action, int):
            if hasattr(self, 'action_mapping') and self.action_mapping and action in self.action_mapping:
                self._execute_action(self.action_mapping[action])
            else:
                action_key = self.action_space[action]
                self._execute_action(action_key)
        else:
            self._execute_action(action)

        # Wait for action to take effect
        time.sleep(self.action_delay)

        # Capture result (with frame skip for performance)
        for _ in range(self.frame_skip - 1):
            time.sleep(0.016)  # ~60 FPS

        screenshot = self._get_screenshot()
        processed = self.preprocess_fn(screenshot)

        # Create GameInfo and extract information
        current_info = GameInfo(
            screenshot=screenshot,
            processed=processed,
            memory_monitor=self.memory_monitor
        )
        current_info = self.extract_info_fn(current_info)

        # Calculate reward
        reward = self.reward_fn(self.previous_info, current_info)

        # Update state
        self.current_state = self.update_state_fn(
            self.current_state, current_info, action
        )

        # Increment step counter
        self.episode_steps += 1

        # Check if episode is done
        done = self._is_episode_done(current_info)

        # Store info for next step
        self.previous_info = current_info

        return processed, reward, done, current_info

    def _get_screenshot(self) -> np.ndarray:
        """Capture screenshot from game."""
        # Try PrintWindow first, if black try desktop BitBlt
        screenshot = self.controller.capture_screenshot(use_printwindow=True)

        # Check if screenshot is mostly black (failed capture)
        if screenshot.mean() < 5:
            # Fallback to desktop capture (requires window to be visible)
            print("PrintWindow capture failed, using desktop BitBlt")
            screenshot = self.controller.capture_screenshot(use_printwindow=False)

        return screenshot

    def _execute_action(self, action: str | dict) -> None:
        """
        Execute action in game.

        Supports:
            - String actions: 'w', 'space', 'left_click', 'right_click'
            - Dict actions for complex operations:
                - {'type': 'key', 'key': 'w'}  # key press
                - {'type': 'key_down', 'key': 'w'}  # key down only
                - {'type': 'key_up', 'key': 'w'}  # key up only
                - {'type': 'mouse_click', 'x': 100, 'y': 200, 'button': 'left'}
                - {'type': 'mouse_down', 'button': 'left'}
                - {'type': 'mouse_up', 'button': 'left'}
                - {'type': 'mouse_move', 'x': 100, 'y': 200}
        """
        if isinstance(action, str):
            # Simple string actions
            if 'click' in action:
                button = 'left' if 'left' in action else 'right'
                self.controller.send_mouse_click(
                    x=self.mouse_position[0],
                    y=self.mouse_position[1],
                    button=button
                )
            elif action.endswith('_down'):
                # e.g., 'w_down' -> key_down('w')
                key = action.replace('_down', '')
                self.controller.key_down(key)
            elif action.endswith('_up'):
                # e.g., 'w_up' -> key_up('w')
                key = action.replace('_up', '')
                self.controller.key_up(key)
            else:
                self.controller.send_key(action)
        elif isinstance(action, dict):
            # Complex dict-based actions
            action_type = action.get('type', 'key')

            if action_type == 'key':
                # Key press (down + up)
                hold_time = action.get('hold_time', 0.05)
                self.controller.send_key(action['key'], hold_time=hold_time)

            elif action_type == 'key_down':
                # Key down only
                self.controller.key_down(action['key'])

            elif action_type == 'key_up':
                # Key up only
                self.controller.key_up(action['key'])

            elif action_type == 'click':
                # Mouse click at specified position
                x = action.get('x', self.mouse_position[0])
                y = action.get('y', self.mouse_position[1])
                button = action.get('button', 'left')
                self.controller.send_mouse_click(x, y, button)
                # Update stored mouse position
                self.mouse_position = (x, y)

            elif action_type == 'mouse_down':
                # Mouse button down only
                x = action.get('x', self.mouse_position[0])
                y = action.get('y', self.mouse_position[1])
                button = action.get('button', 'left')
                self.controller.mouse_down(x, y, button)
                self.mouse_position = (x, y)

            elif action_type == 'mouse_up':
                # Mouse button up only
                x = action.get('x', self.mouse_position[0])
                y = action.get('y', self.mouse_position[1])
                button = action.get('button', 'left')
                self.controller.mouse_up(x, y, button)
                self.mouse_position = (x, y)

            elif action_type == 'mouse_move':
                # Move mouse without clicking
                x = action['x']
                y = action['y']
                self.controller.mouse_move(x, y)
                self.mouse_position = (x, y)

            else:
                raise NotImplementedError(f"Unknown action type: {action_type}")

    def _is_episode_done(self, info: GameInfo) -> bool:
        """
        Determine if episode is finished.

        For continuous games, episodes might end when:
        - Player dies / game over
        - Reached certain objective
        - Hit episode timeout

        Args:
            info: Current GameInfo

        Returns:
            True if episode should end
        """
        # Check game-specific conditions (using GameInfo properties)
        if info.game_over:
            return True

        if info.mission_complete:
            return True

        # Check timeout
        if self.episode_timeout and self.episode_steps >= self.episode_timeout:
            return True

        return False

    def close(self) -> None:
        """Clean up resources."""
        if self.memory_monitor:
            self.memory_monitor.close()

    def get_memory_monitor(self) -> Optional[GameMemoryMonitor]:
        """Get memory monitor for manual registration of addresses."""
        return self.memory_monitor

    def set_mouse_position(self, x: int, y: int) -> None:
        """Set the stored mouse position for subsequent click actions."""
        self.mouse_position = (x, y)

    def get_mouse_position(self) -> tuple[int, int]:
        """Get the current stored mouse position."""
        return self.mouse_position

    def set_action_mapping(self, action_mapping: dict[int, dict]) -> None:
        """
        Set action mapping for complex actions (e.g., grid clicks).

        Args:
            action_mapping: Mapping from action index to dict action
        """
        self.action_mapping = action_mapping

    def execute_mapped_action(self, action_idx: int, action_mapping: Optional[dict[int, dict]] = None) -> None:
        """
        Execute an action that may be mapped to a complex operation.

        Args:
            action_idx: Index into action_space
            action_mapping: Optional mapping from action index to dict action
                          (for grid clicks, etc.)
        """
        if action_mapping and action_idx in action_mapping:
            # Use mapped dict action (e.g., grid click with coordinates)
            self._execute_action(action_mapping[action_idx])
        else:
            # Use simple string action from action_space
            action_key = self.action_space[action_idx]
            self._execute_action(action_key)
