"""
Configuration management module.
Supports loading configs from YAML/JSON files.
"""
import json
import yaml
from typing import Any, Optional
from pathlib import Path


class Config:
    """Configuration manager"""

    def __init__(self, config_dict: Optional[dict[str, Any]] = None):
        """
        Initialize config.

        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict or {}

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value"""
        keys = key.split('.')
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set config value"""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return self._config.copy()

    def save_json(self, path: str) -> None:
        """Save config to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)

    def save_yaml(self, path: str) -> None:
        """Save config to YAML file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)


# Example default configuration
DEFAULT_CONFIG = {
    'game': {
        'window_name': 'YourGameName',
        'action_space': ['w', 'a', 's', 'd', 'space'],
        'frame_skip': 4,
        'action_delay': 0.05
    },
    'model': {
        'type': 'DQN',
        'state_shape': [84, 84, 1],
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'update_target_every': 1000
    },
    'training': {
        'num_episodes': 1000,
        'max_steps_per_episode': 500,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'buffer_size': 50000,
        'update_frequency': 4,
        'save_frequency': 50,
        'checkpoint_path': 'checkpoints/model.pth'
    },
    'preprocessing': {
        'resize': [84, 84],
        'grayscale': True,
        'normalize': True
    }
}


def create_default_config(path: str, format: str = 'yaml') -> None:
    """
    Create a default configuration file.

    Args:
        path: Path to save config file
        format: 'json' or 'yaml'
    """
    config = Config(DEFAULT_CONFIG)

    if format == 'json':
        config.save_json(path)
    else:
        config.save_yaml(path)
