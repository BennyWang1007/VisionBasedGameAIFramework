"""
Example script for playing/testing a trained model.
"""
import logging
# from src.core import GameInfo
from src.environment import ContinuousGameEnvironment
from src.models import DQNModel
from src.agent import RLAgent
from examples.train_example import (
    preprocess_screenshot,
    extract_game_info,
    calculate_reward,
    update_state
)


def main():
    """Play with trained model"""

    logging.basicConfig(level=logging.INFO)

    # Configuration (must match training config)
    WINDOW_NAME = "YourGameName"  # TODO: Set your game window name
    ACTION_SPACE = ['w', 'a', 's', 'd', 'space']
    CHECKPOINT_PATH = 'checkpoints/dqn_model.pth'

    model_config = {
        'state_shape': (84, 84, 1),
        'num_actions': len(ACTION_SPACE),
        'learning_rate': 0.00025,
        'gamma': 0.99,
    }

    agent_config = {
        'gamma': 0.99,
        'batch_size': 32,
        'buffer_size': 10000,
    }

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

    # Create model and agent
    model = DQNModel(model_config)
    agent = RLAgent(model, env, agent_config)

    # Play
    print("Loading model and starting play mode...")
    print("Press Ctrl+C to stop")
    agent.play(checkpoint_path=CHECKPOINT_PATH, delay=0.1)


if __name__ == '__main__':
    main()
