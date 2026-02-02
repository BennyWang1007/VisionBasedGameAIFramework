"""
Example implementation showing how to use the framework.
You should customize the callback functions for your specific game.
"""
import numpy as np
import cv2
from typing import Any

from src.core import GameInfo
from src.environment import ContinuousGameEnvironment
from src.models import DQNModel
from src.agent import RLAgent
import logging


# ============================================================================
# CALLBACK FUNCTIONS - CUSTOMIZE THESE FOR YOUR GAME
# ============================================================================

def preprocess_screenshot(screenshot: np.ndarray) -> np.ndarray:
    """
    Preprocess raw screenshot for model input.

    Args:
        screenshot: Raw screenshot (H, W, 3) RGB

    Returns:
        Preprocessed state (e.g., 84x84 grayscale)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)

    # Resize to standard size
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    # Normalize
    normalized = resized.astype(np.float32) / 255.0

    # Add channel dimension
    return np.expand_dims(normalized, axis=-1)


def extract_game_info(game_info: GameInfo) -> GameInfo:
    """
    Extract game information from screenshot.
    This could involve OCR, template matching, color detection, etc.

    Args:
        game_info: GameInfo with screenshot, processed frame, and memory_monitor

    Returns:
        GameInfo with populated data dictionary
    """
    # PLACEHOLDER - Implement your game-specific extraction logic
    # Examples:
    # - OCR to read score, health, ammo
    # - Template matching to detect enemies
    # - Color detection to find objectives
    # - Object detection for game elements
    #
    # You can access:
    # - game_info.screenshot: Raw screenshot
    # - game_info.processed: Preprocessed observation
    # - game_info.memory_monitor: Memory reader (if enabled)

    game_info.data = {
        'score': 0,  # TODO: Extract actual score
        'health': 100,  # TODO: Extract health
        'enemies_visible': 0,  # TODO: Detect enemies
        'position_x': 0,  # TODO: Extract position
        'position_y': 0,
        'game_over': False,  # TODO: Detect game over
    }
    return game_info


def calculate_reward(prev_info: GameInfo, current_info: GameInfo) -> float:
    """
    Calculate reward based on game state changes.

    Args:
        prev_info: Previous GameInfo
        current_info: Current GameInfo

    Returns:
        Reward value
    """
    reward = 0.0

    # Example reward structure - customize for your game

    # Reward for score increase
    if current_info.get('score', 0) > prev_info.get('score', 0):
        reward += (current_info.get('score', 0) - prev_info.get('score', 0)) * 10

    # Penalty for health decrease
    if current_info.get('health', 100) < prev_info.get('health', 100):
        reward -= (prev_info.get('health', 100) - current_info.get('health', 100)) * 5

    # Penalty for game over
    if current_info.game_over:
        reward -= 100

    # Small reward for survival
    reward += 0.1

    return reward


def update_state(current_state: dict, new_info: GameInfo, action: Any) -> dict:
    """
    Update internal state representation.
    This can maintain a history or derived features.

    Args:
        current_state: Current state dictionary
        new_info: New GameInfo from game
        action: Action that was taken

    Returns:
        Updated state dictionary
    """
    # Example: Maintain action history
    action_history = current_state.get('action_history', [])
    if action is not None:
        action_history.append(action)
        # Keep only last 10 actions
        action_history = action_history[-10:]

    return {
        'info': new_info.data,
        'action_history': action_history,
        'steps': current_state.get('steps', 0) + 1
    }


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    # Game configuration
    WINDOW_NAME = "YourGameName"  # TODO: Set your game window name
    ACTION_SPACE = ['w', 'a', 's', 'd', 'space']  # TODO: Define your actions

    # Model configuration
    model_config = {
        'state_shape': (84, 84, 1),  # Matches preprocessed screenshot shape
        'num_actions': len(ACTION_SPACE),
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'update_target_every': 1000
    }

    # Agent configuration
    agent_config = {
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'buffer_size': 50000,
        'update_frequency': 4
    }

    # Training configuration
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 500
    SAVE_FREQUENCY = 50
    CHECKPOINT_PATH = 'checkpoints/dqn_model.pth'

    # ========================================================================
    # SETUP
    # ========================================================================

    # Create environment
    env = ContinuousGameEnvironment(
        window_name=WINDOW_NAME,
        preprocess_fn=preprocess_screenshot,
        extract_info_fn=extract_game_info,
        reward_fn=calculate_reward,
        update_state_fn=update_state,
        action_space=ACTION_SPACE,
        frame_skip=4,
        action_delay=0.05
    )

    # Create model
    model = DQNModel(model_config)

    # Create agent
    agent = RLAgent(model, env, agent_config)

    # ========================================================================
    # TRAINING
    # ========================================================================

    print("Starting training...")
    print("Model: DQN")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Action space: {ACTION_SPACE}")
    print("-" * 50)

    agent.train(
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        save_frequency=SAVE_FREQUENCY,
        checkpoint_path=CHECKPOINT_PATH
    )

    print("Training completed!")

    # ========================================================================
    # EVALUATION
    # ========================================================================

    print("\nEvaluating model...")
    metrics = agent.evaluate(num_episodes=10)

    print("\nEvaluation Results:")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Steps: {metrics['mean_steps']:.2f} ± {metrics['std_steps']:.2f}")


if __name__ == '__main__':
    main()
