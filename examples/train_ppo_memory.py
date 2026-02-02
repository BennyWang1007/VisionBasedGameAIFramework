"""
Example implementation using PPO with memory reading for continuous games.
This shows how to use memory addresses to extract game state.
"""
import numpy as np
import cv2
from typing import Any
import logging

from src.core import GameInfo
from src.environment import ContinuousGameEnvironment
from src.models import PPOModel, compute_gae
from src.utils import GameMemoryMonitor


# ============================================================================
# CALLBACK FUNCTIONS WITH MEMORY READING
# ============================================================================

def preprocess_screenshot(screenshot: np.ndarray) -> np.ndarray:
    """
    Preprocess raw screenshot for model input.

    Args:
        screenshot: Raw screenshot (H, W, 3) RGB

    Returns:
        Preprocessed state (84x84 grayscale)
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
    Extract game information using memory reading.

    This is where you read memory addresses to get exact game state.
    Much more reliable than computer vision for internal state!

    Args:
        game_info: GameInfo with screenshot, processed frame, and memory_monitor

    Returns:
        GameInfo with populated data dictionary
    """
    if game_info.memory_monitor is None:
        # Fallback if memory reading is disabled
        game_info.data = {
            'score': 0,
            'health': 100,
            'ammo': 100,
            'position_x': 0.0,
            'position_y': 0.0,
            'game_over': False,
        }
        return game_info

    # Read all registered values from memory
    # These addresses should be registered in setup_memory_addresses()
    memory_values = game_info.memory_monitor.read_all()
    game_info.data.update(memory_values)

    # Add any computed values
    game_info.data['health_percent'] = (game_info.get('health', 0) / game_info.get('max_health', 100)) * 100
    game_info.data['ammo_percent'] = (game_info.get('ammo', 0) / game_info.get('max_ammo', 100)) * 100

    # Check game over condition
    game_info.data['game_over'] = game_info.get('health', 100) <= 0

    return game_info


def calculate_reward(prev_info: GameInfo, current_info: GameInfo) -> float:
    """
    Calculate reward based on game state changes from memory.

    Args:
        prev_info: Previous GameInfo
        current_info: Current GameInfo

    Returns:
        Reward value
    """
    reward = 0.0

    # Reward for score increase
    score_diff = current_info.get('score', 0) - prev_info.get('score', 0)
    if score_diff > 0:
        reward += score_diff * 1.0

    # Penalty for health loss
    health_diff = current_info.get('health', 100) - prev_info.get('health', 100)
    if health_diff < 0:
        reward += health_diff * 0.1  # Negative value

    # Reward for getting closer to objective (if position-based)
    # Example: Reward for moving toward target
    target_x, target_y = current_info.get('target_x', 0), current_info.get('target_y', 0)
    prev_x, prev_y = prev_info.get('position_x', 0), prev_info.get('position_y', 0)
    curr_x, curr_y = current_info.get('position_x', 0), current_info.get('position_y', 0)

    prev_dist = np.sqrt((target_x - prev_x)**2 + (target_y - prev_y)**2)
    curr_dist = np.sqrt((target_x - curr_x)**2 + (target_y - curr_y)**2)

    if prev_dist > 0:  # Avoid division by zero
        reward += (prev_dist - curr_dist) * 0.01  # Small reward for getting closer

    # Large penalty for game over
    if current_info.game_over:
        reward -= 10.0

    # Small survival reward
    reward += 0.01

    return reward


def update_state(current_state: dict, new_info: GameInfo, action: Any) -> dict:
    """
    Update internal state representation.

    Args:
        current_state: Current state dictionary
        new_info: New GameInfo from game
        action: Action that was taken

    Returns:
        Updated state dictionary
    """
    # Maintain action history
    action_history = current_state.get('action_history', [])
    if action is not None:
        action_history.append(action)
        action_history = action_history[-10:]  # Keep last 10

    # Track health changes for pattern detection
    health_history = current_state.get('health_history', [])
    health_history.append(new_info.get('health', 100))
    health_history = health_history[-5:]  # Keep last 5

    return {
        'info': new_info.data,
        'action_history': action_history,
        'health_history': health_history,
        'steps': current_state.get('steps', 0) + 1
    }


def setup_memory_addresses(memory_monitor: GameMemoryMonitor):
    """
    Register all memory addresses you want to monitor.

    You need to find these addresses using tools like:
    - Cheat Engine
    - Process Hacker
    - x64dbg
    - Game-specific memory scanners

    Args:
        memory_monitor: Memory monitor to register addresses with
    """
    # TODO: Replace with your actual game addresses!

    # Example: Simple static addresses
    memory_monitor.register_address('health', 0x12345678, 'int')
    memory_monitor.register_address('max_health', 0x1234567C, 'int')
    memory_monitor.register_address('score', 0x12345680, 'int')
    memory_monitor.register_address('ammo', 0x12345684, 'int')
    memory_monitor.register_address('max_ammo', 0x12345688, 'int')

    # Example: Float positions
    memory_monitor.register_address('position_x', 0x1234568C, 'float')
    memory_monitor.register_address('position_y', 0x12345690, 'float')
    memory_monitor.register_address('position_z', 0x12345694, 'float')

    # Example: Pointer chains (for dynamic addresses)
    # Format: base address, then list of offsets
    # This follows: [[base + offset1] + offset2] + offset3
    base_address = 0x00400000  # Your game's base address
    memory_monitor.register_pointer_chain(
        'player_health',
        base_address,
        [0x12A4F0, 0x28, 0x10, 0x4],  # Offsets to follow
        'int'
    )

    memory_monitor.register_pointer_chain(
        'target_x',
        base_address,
        [0x12A500, 0x30],
        'float'
    )

    memory_monitor.register_pointer_chain(
        'target_y',
        base_address,
        [0x12A500, 0x34],
        'float'
    )

    print("Memory addresses registered!")
    print("Note: These are placeholder addresses. Use Cheat Engine to find real ones!")


# ============================================================================
# PPO TRAINING FOR CONTINUOUS GAMES
# ============================================================================

class PPOTrainer:
    """PPO trainer for continuous games"""

    def __init__(self, env, model, config):
        self.env = env
        self.model = model
        self.config = config

        self.logger = logging.getLogger(__name__)

    def collect_rollout(self, num_steps: int):
        """
        Collect a rollout of experiences.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Dictionary with collected data
        """
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        state = self.env.reset()

        for _ in range(num_steps):
            # Get action from policy
            action, log_prob, value = self.model.get_action(state)

            # Execute action
            next_state, reward, done, info = self.env.step(action)

            # Store transition
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            state = next_state

            if done:
                state = self.env.reset()

        # Get final value for bootstrapping
        _, _, final_value = self.model.get_action(state)

        # Compute advantages and returns using GAE
        advantages, returns = compute_gae(
            np.array(rewards),
            np.array(values),
            np.array([final_value]),
            np.array(dones),
            self.config['gamma'],
            self.config.get('gae_lambda', 0.95)
        )

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'old_log_probs': np.array(log_probs),
            'rewards': np.array(rewards),
            'returns': returns,
            'advantages': advantages,
            'values': np.array(values)
        }

    def train(self, num_iterations: int, steps_per_iteration: int):
        """
        Train the agent.

        Args:
            num_iterations: Number of training iterations
            steps_per_iteration: Steps to collect per iteration
        """
        for iteration in range(num_iterations):
            # Collect rollout
            batch = self.collect_rollout(steps_per_iteration)

            # Train on collected data
            metrics = self.model.train_step(batch)

            # Log results
            mean_reward = batch['rewards'].mean()
            mean_return = batch['returns'].mean()

            self.logger.info(
                f"Iteration {iteration + 1}/{num_iterations} - "
                f"Mean Reward: {mean_reward:.2f}, "
                f"Mean Return: {mean_return:.2f}, "
                f"Policy Loss: {metrics['policy_loss']:.4f}, "
                f"Value Loss: {metrics['value_loss']:.4f}"
            )

            # Save checkpoint
            if (iteration + 1) % 10 == 0:
                self.model.save(f'checkpoints/ppo_iteration_{iteration + 1}.pth')


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

    WINDOW_NAME = "Hades II"  # TODO: Your game window name
    PROCESS_NAME = "Hades2.exe"     # TODO: Your game process name
    ACTION_SPACE = ['w', 'a', 's', 'd', 'space', 'left_click']  # TODO: Your actions

    # PPO model configuration
    model_config = {
        'state_shape': (84, 84, 1),
        'num_actions': len(ACTION_SPACE),
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'ppo_epochs': 4,
        'minibatch_size': 64,
    }

    # Training configuration
    train_config = {
        'num_iterations': 1000,
        'steps_per_iteration': 2048,
        'gamma': 0.99,
        'gae_lambda': 0.95,
    }

    # ========================================================================
    # SETUP
    # ========================================================================

    # Create environment
    env = ContinuousGameEnvironment(
        window_name=WINDOW_NAME,
        process_name=PROCESS_NAME,
        preprocess_fn=preprocess_screenshot,
        extract_info_fn=extract_game_info,
        reward_fn=calculate_reward,
        update_state_fn=update_state,
        action_space=ACTION_SPACE,
        use_memory_reading=True,
        frame_skip=4,
        action_delay=0.05,
        episode_timeout=1000  # Max steps per episode
    )

    # Setup memory addresses
    memory_monitor = env.get_memory_monitor()
    if memory_monitor:
        setup_memory_addresses(memory_monitor)
    else:
        print("Warning: Memory reading is disabled!")

    # Create PPO model
    model = PPOModel(model_config)

    # Create trainer
    trainer = PPOTrainer(env, model, train_config)

    # ========================================================================
    # TRAINING
    # ========================================================================

    print("Starting PPO training for continuous game...")
    print("Model: PPO")
    print(f"Iterations: {train_config['num_iterations']}")
    print(f"Steps per iteration: {train_config['steps_per_iteration']}")
    print(f"Action space: {ACTION_SPACE}")
    print("-" * 50)

    trainer.train(
        num_iterations=train_config['num_iterations'],
        steps_per_iteration=train_config['steps_per_iteration']
    )

    print("Training completed!")
    env.close()


if __name__ == '__main__':
    main()
