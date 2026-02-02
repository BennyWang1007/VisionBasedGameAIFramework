"""
RL Agent that orchestrates training and inference.
"""
import numpy as np
from typing import Optional, Any
import time
import logging

from src.core import BaseModel, BaseEnvironment, ReplayBuffer


class RLAgent:
    """Main RL agent that handles training loop"""

    def __init__(
        self,
        model: BaseModel,
        environment: BaseEnvironment,
        config: dict[str, Any]
    ):
        """
        Initialize RL agent.

        Args:
            model: RL model instance
            environment: Game environment instance
            config: Configuration dictionary
        """
        self.model = model
        self.env = environment
        self.config = config

        # Training parameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.update_frequency = config.get('update_frequency', 4)

        # Replay buffer
        buffer_size = config.get('buffer_size', 10000)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Training state
        self.epsilon = self.epsilon_start
        self.total_steps = 0
        self.episode_count = 0

        # Logging
        self.logger = logging.getLogger(__name__)

    def train(
        self,
        num_episodes: int,
        max_steps_per_episode: Optional[int] = None,
        render: bool = False,
        save_frequency: int = 100,
        checkpoint_path: str = 'checkpoints/model.pth'
    ) -> None:
        """
        Train the agent.

        Args:
            num_episodes: Number of episodes to train
            max_steps_per_episode: Maximum steps per episode (None for unlimited)
            render: Whether to render during training
            save_frequency: Save model every N episodes
            checkpoint_path: Path to save checkpoints
        """
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward: float = 0.0
            episode_steps = 0
            done = False

            while not done:
                # Select action
                action = self.model.get_action(state, epsilon=self.epsilon)

                # Execute action
                next_state, reward, done, info = self.env.step(action)

                # Store transition
                self.replay_buffer.add(state, action, reward, next_state, done)

                # Update metrics
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1

                # Train model
                if len(self.replay_buffer) >= self.batch_size and \
                   self.total_steps % self.update_frequency == 0:
                    batch = self.replay_buffer.sample(self.batch_size)
                    metrics = self.model.train_step(batch)

                    # Log training metrics
                    if self.total_steps % 100 == 0:
                        self.logger.info(f"Step {self.total_steps}: {metrics}")

                state = next_state

                # Check max steps
                if max_steps_per_episode and episode_steps >= max_steps_per_episode:
                    break

            # Update epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.episode_count += 1

            # Log episode results
            self.logger.info(
                f"Episode {episode + 1}/{num_episodes} - "
                f"Reward: {episode_reward:.2f}, "
                f"Steps: {episode_steps}, "
                f"Epsilon: {self.epsilon:.4f}"
            )

            # Save checkpoint
            if (episode + 1) % save_frequency == 0:
                self.model.save(checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Final save
        self.model.save(checkpoint_path)
        self.env.close()

    def evaluate(
        self,
        num_episodes: int = 10,
        max_steps_per_episode: Optional[int] = None,
        render: bool = True
    ) -> dict[str, float]:
        """
        Evaluate the trained agent.

        Args:
            num_episodes: Number of episodes to evaluate
            max_steps_per_episode: Maximum steps per episode
            render: Whether to render during evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        total_rewards = []
        total_steps_list = []

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                # Select action (no exploration)
                action = self.model.get_action(state, epsilon=0.0)

                # Execute action
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1
                state = next_state

                if max_steps_per_episode and episode_steps >= max_steps_per_episode:
                    break

            total_rewards.append(episode_reward)
            total_steps_list.append(episode_steps)

            self.logger.info(
                f"Evaluation Episode {episode + 1}/{num_episodes} - "
                f"Reward: {episode_reward:.2f}, Steps: {episode_steps}"
            )

        self.env.close()

        return {
            'mean_reward': float(np.mean(total_rewards)),
            'std_reward': float(np.std(total_rewards)),
            'mean_steps': float(np.mean(total_steps_list)),
            'std_steps': float(np.std(total_steps_list))
        }

    def play(self, checkpoint_path: str, delay: float = 0.1) -> None:
        """
        Load model and play indefinitely (for demonstration).

        Args:
            checkpoint_path: Path to model checkpoint
            delay: Delay between actions (seconds)
        """
        self.model.load(checkpoint_path)

        try:
            while True:
                state = self.env.reset()
                done = False
                episode_reward: float = 0.0

                while not done:
                    action = self.model.get_action(state, epsilon=0.0)
                    next_state, reward, done, info = self.env.step(action)

                    episode_reward += reward
                    state = next_state

                    time.sleep(delay)

                self.logger.info(f"Episode finished - Reward: {episode_reward:.2f}")

        except KeyboardInterrupt:
            self.logger.info("Stopped by user")
        finally:
            self.env.close()
