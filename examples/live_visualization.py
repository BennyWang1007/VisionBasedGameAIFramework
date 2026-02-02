"""
Live visualization example using real game data with PPO.
Captures game screenshots, preprocesses them, and visualizes in real-time.
Sends actions to the game.
"""
import numpy as np
import cv2
import time
import logging
from typing import Optional

from src.core import GameInfo
from src.environment import ContinuousGameEnvironment
from src.models import PPOModel
# from src.utils import GameMemoryMonitor

# Import visualization GUI
from examples.visualization_gui import VisualizationGUI

# Reuse setup from train_ppo_memory
from examples.train_ppo_memory import (
    setup_memory_addresses,
)


def preprocess_screenshot(screenshot: np.ndarray) -> np.ndarray:
    """
    Preprocess raw screenshot using Canny edge detection.

    Args:
        screenshot: Raw screenshot (H, W, 3) RGB

    Returns:
        Canny edge image (84, 84, 1) normalized
    """
    # Convert to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Resize to 84x84
    # resized = cv2.resize(edges, (84, 84), interpolation=cv2.INTER_AREA)

    # Normalize to 0-1
    # normalized = resized.astype(np.float32) / 255.0
    normalized = edges.astype(np.float32) / 255.0

    # Add channel dimension
    return np.expand_dims(normalized, axis=-1)


def extract_game_info(game_info: GameInfo) -> GameInfo:
    """
    Extract game info - stores the resized 84x84 grayscale image.

    Args:
        game_info: GameInfo with screenshot and processed frame

    Returns:
        GameInfo with resized image in data
    """
    # Resize the screenshot to 84x84 grayscale for model input

    if game_info.processed is not None:
        resized = cv2.resize(
            game_info.processed.squeeze(), (84, 84), interpolation=cv2.INTER_AREA
        )
        normalized = resized.astype(np.float32) / 255.0
        game_info.data['resized_frame'] = np.expand_dims(normalized, axis=-1)

    elif game_info.screenshot is not None:
        gray = cv2.cvtColor(game_info.screenshot, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        game_info.data['resized_frame'] = np.expand_dims(normalized, axis=-1)

    # Placeholder game state data
    game_info.data.update({
        'score': np.random.randint(0, 10000),
        'health': np.random.uniform(0, 100),
        'max_health': 100,
        'game_over': False,
    })

    return game_info


# ============================================================================
# SIMPLIFIED CALLBACKS FOR VISUALIZATION
# ============================================================================

def extract_game_info_visual(game_info: GameInfo) -> GameInfo:
    """
    Extract game information for visualization.
    Uses random reward since we're focusing on visualization.

    Args:
        game_info: GameInfo with screenshot and processed frame

    Returns:
        GameInfo with populated data dictionary
    """
    # Store the processed image in data for easy access
    game_info.data['processed_frame'] = game_info.processed

    if game_info.memory_monitor is None:
        # Fallback with placeholder data
        game_info.data.update({
            'score': np.random.randint(0, 10000),
            'health': np.random.uniform(0, 100),
            'max_health': 100,
            'ammo': np.random.randint(0, 100),
            'position_x': np.random.uniform(0, 1000),
            'position_y': np.random.uniform(0, 1000),
            'game_over': False,
        })
    else:
        # Read all registered values from memory
        memory_values = game_info.memory_monitor.read_all()
        game_info.data.update(memory_values)

        # Add computed values
        game_info.data['health_percent'] = (
            game_info.get('health', 0) / game_info.get('max_health', 100)
        ) * 100
        game_info.data['game_over'] = game_info.get('health', 100) <= 0

    return game_info


def calculate_reward_random(prev_info: GameInfo, current_info: GameInfo) -> float:
    """
    Random reward for visualization demo.
    Replace with actual reward function for training.
    """
    return np.random.uniform(-1, 1)


def update_state_simple(current_state: dict, new_info: GameInfo, action) -> dict:
    """Simple state update for visualization."""
    return {
        'info': new_info.data,
        'steps': current_state.get('steps', 0) + 1,
        'last_action': action,
    }


# ============================================================================
# LIVE VISUALIZATION CLASS
# ============================================================================

class LiveGameVisualizer:
    """
    Real-time game visualization with PPO model.
    Captures game screen, preprocesses, runs inference, and sends actions.
    """

    def __init__(
        self,
        window_name: str,
        process_name: str,
        action_space: list[str],
        model_path: Optional[str] = None,
        use_memory_reading: bool = False,
        action_delay: float = 0.1,
    ):
        """
        Initialize the live visualizer.

        Args:
            window_name: Name of the game window to capture
            process_name: Process name for memory reading
            action_space: List of action keys
            model_path: Path to trained PPO model (None for random actions)
            use_memory_reading: Whether to use memory reading for game state
            action_delay: Delay between actions in seconds
        """
        self.action_space = action_space
        self.action_delay = action_delay
        self.running = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create environment
        self.env = ContinuousGameEnvironment(
            window_name=window_name,
            process_name=process_name,
            preprocess_fn=preprocess_screenshot,  # Reused from train_ppo_memory
            extract_info_fn=extract_game_info,
            reward_fn=calculate_reward_random,
            update_state_fn=update_state_simple,
            action_space=action_space,
            use_memory_reading=use_memory_reading,
            frame_skip=1,  # No frame skip for visualization
            action_delay=action_delay,
        )

        # Setup memory addresses if using memory reading
        if use_memory_reading:
            memory_monitor = self.env.get_memory_monitor()
            if memory_monitor:
                setup_memory_addresses(memory_monitor)

        # Model configuration
        model_config = {
            'state_shape': (84, 84, 1),
            'num_actions': len(action_space),
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'ppo_epochs': 4,
            'minibatch_size': 64,
        }

        # Create PPO model
        self.model = PPOModel(model_config)

        # Load trained weights if provided
        if model_path:
            self.model.load(model_path)
            self.logger.info(f"Loaded model from {model_path}")
        else:
            self.logger.info("Using random policy (no model loaded)")

        # Create visualization GUI
        self.gui = VisualizationGUI(width=1200, height=800)

        # Statistics
        self.total_steps = 0
        self.episode_reward = 0.0

        # Track current state for decision making
        self.current_processed = None
        self.current_info = None

        # Create action mapping for complex actions (clicks, mouse moves)
        self.action_mapping = self.get_action_mapping()

    def _get_action_with_confidence(self, state: np.ndarray) -> tuple[int, float]:
        """
        Get action from model with confidence score.

        Args:
            state: Preprocessed state (84, 84, 1)

        Returns:
            (action_index, confidence)
        """
        action, log_prob, value = self.model.get_action(state)

        # Convert log_prob to confidence (probability)
        confidence = np.exp(log_prob)

        return int(action), float(confidence)

    def get_action_mapping(self) -> dict[int, dict]:
        """
        Create action mapping for complex actions (e.g., grid clicks, mouse moves).

        Returns:
            Mapping from action index to dict action
        """
        action_mapping = {}
        rect = self.env.controller.get_window_rect()
        if rect:
            screen_width = rect[2] - rect[0]
            screen_height = rect[3] - rect[1]
        else:
            screen_width = 800
            screen_height = 600
        dxdy = 100

        for idx, action_key in enumerate(self.action_space):
            parts = action_key.split('_')

            # Handle mouse click/down/up actions
            if parts[0] == 'left' or parts[0] == 'right':
                button = parts[0]
                action_type = 'mouse_' + parts[1]  # click, down, up
                action_mapping[idx] = {
                    'type': action_type,
                    'button': button,
                }

            # Handle mouse move actions
            elif parts[0] == 'mouse' and parts[1] == 'move':
                # Format: mouse_move_row_col (e.g., mouse_move_0_0)
                row = int(parts[2])
                col = int(parts[3])
                x = int(screen_width / 2) + (col - 1) * dxdy
                y = int(screen_height / 2) + (row - 1) * dxdy
                action_mapping[idx] = {
                    'type': 'mouse_move',
                    'x': x,
                    'y': y,
                }

            # Handle key presses
            elif len(parts) == 1:
                # Simple key press
                action_mapping[idx] = {
                    'type': 'key',
                    'key': action_key,
                    'hold_time': 0.05,
                }

            # Handle key down/up
            elif len(parts) == 2:
                if parts[1] == 'down' or parts[1] == 'up':
                    # Key down or up
                    action_mapping[idx] = {
                        'type': 'key_' + parts[1],
                        'key': parts[0],
                    }

            if idx not in action_mapping:
                raise ValueError(f"Unknown action key format: {action_key}")

        return action_mapping

    def run_step(self) -> bool:
        """
        Run a single step: capture, process, act, visualize.

        Returns:
            True if should continue, False if done
        """
        try:
            # Initialize state if this is the first step
            if self.current_info is None:
                self.current_processed = self.env.reset()
                self.current_info = self.env.previous_info
                # Extract info which creates the resized_frame for model
                self.current_info = extract_game_info(self.current_info)

            # Get the resized frame (84x84) for model input
            model_input = self.current_info.data.get('resized_frame', self.current_processed)

            # Get action from model based on current state
            action_idx, confidence = self._get_action_with_confidence(model_input)
            action_name = self.action_space[action_idx]

            # Execute action through environment step and get next state
            action = self.action_mapping.get(action_idx, action_name)
            next_processed, reward, done, next_info = self.env.step(action)

            # Extract info (creates resized_frame) and visual info
            next_info = extract_game_info(next_info)
            game_info = extract_game_info_visual(next_info)

            # Update statistics
            self.episode_reward += reward
            self.total_steps += 1

            # Update GUI with next state (result of action)
            self.gui.update_screenshot(next_info.screenshot)
            self.gui.update_processed_screen(next_processed)

            # Add step info to game states
            display_states = game_info.data.copy()
            display_states['step'] = self.total_steps
            display_states['episode_reward'] = round(self.episode_reward, 2)
            display_states['reward'] = round(reward, 3)
            self.gui.update_game_states(display_states)

            self.gui.update_action(action_name, action_id=action_idx, confidence=confidence)

            # Process GUI events
            self.gui.update()

            # Store next state as current for next iteration
            self.current_processed = next_processed
            self.current_info = next_info

            return not done

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error in run_step: {error_msg}")

            # Stop if GUI window was closed
            if "invalid command name " in error_msg and "!canvas" in error_msg:
                self.logger.info("GUI window closed, stopping...")
                self.running = False
                return False

            return False

    def run(self):
        """Run the visualization loop."""
        self.running = True
        self.logger.info("Starting live visualization...")
        self.logger.info(f"Action space: {self.action_space}")
        self.logger.info("Press Ctrl+C to stop")

        try:
            while self.running:
                if not self.run_step():
                    if not self.running:
                        # GUI was closed, exit cleanly
                        break
                    self.logger.info("Episode ended, restarting...")
                    self.episode_reward = 0.0
                    time.sleep(1.0)  # Brief pause before restart

        except KeyboardInterrupt:
            self.logger.info("Stopped by user")
        finally:
            self.close()

    def close(self):
        """Clean up resources."""
        self.running = False
        self.env.close()
        try:
            self.gui.close()
        except Exception:
            pass  # GUI might already be closed
        self.logger.info("Visualization closed")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run live game visualization."""

    # ========================================================================
    # CONFIGURATION - Modify these for your game!
    # ========================================================================

    WINDOW_NAME = "Hades II"          # Your game window name
    PROCESS_NAME = "Hades2.exe"       # Your game process name
    ACTION_SPACE = [                   # Your action keys
        'w',            # Move up
        'a',            # Move left
        's',            # Move down
        'd',            # Move right
        'space',        # Dash/Dodge
        'left_click',   # Attack
        'left_down',    # Hold left mouse button (for charged attack)
        'left_up',      # Release left mouse button
        'right_click',  # Special Attack
        'right_down',   # Hold right mouse button
        'right_up',     # Release right mouse button
        'e',            # Cast
        'e_down',        # Hold 'e' (for cast)
        'e_up',         # Release 'e'
        'mouse_move_0_0',   # Move mouse to left-top of character
        'mouse_move_0_1',   # Move mouse to left-middle
        'mouse_move_0_2',   # Move mouse to left-bottom
        'mouse_move_1_0',   # Move mouse to center-top
        'mouse_move_1_1',   # Move mouse to center
        'mouse_move_1_2',   # Move mouse to center-bottom
        'mouse_move_2_0',   # Move mouse to right-top
        'mouse_move_2_1',   # Move mouse to right-middle
        'mouse_move_2_2',   # Move mouse to right-bottom
    ]

    # Optional: Path to trained model
    MODEL_PATH = None  # Set to 'checkpoints/ppo_model.pth' if you have a trained model

    # Use memory reading for accurate game state
    USE_MEMORY_READING = False  # Set to True if you have memory addresses configured

    # Action delay (seconds between actions)
    ACTION_DELAY = 0.1

    # ========================================================================
    # RUN VISUALIZER
    # ========================================================================

    print("=" * 60)
    print("  RL Game AI - Live Visualization")
    print("=" * 60)
    print(f"  Window: {WINDOW_NAME}")
    print(f"  Actions: {ACTION_SPACE}")
    print(f"  Memory Reading: {'Enabled' if USE_MEMORY_READING else 'Disabled'}")
    print(f"  Model: {MODEL_PATH if MODEL_PATH else 'Random Policy'}")
    print("=" * 60)
    print("\nMake sure the game window is visible!")
    print("Press Ctrl+C to stop.\n")

    try:
        visualizer = LiveGameVisualizer(
            window_name=WINDOW_NAME,
            process_name=PROCESS_NAME,
            action_space=ACTION_SPACE,
            model_path=MODEL_PATH,
            use_memory_reading=USE_MEMORY_READING,
            action_delay=ACTION_DELAY,
        )

        visualizer.run()

    except RuntimeError as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print(f"  1. The game '{WINDOW_NAME}' is running")
        print("  2. The window is visible (not minimized)")
        print("  3. The window name matches exactly")


if __name__ == '__main__':
    main()
