# RL-based Game AI

A flexible and modular framework for training reinforcement learning agents to play PC games using Windows API for screen capture and input control.

## Features

- **Modular Architecture**: Clean separation of concerns with abstract base classes
- **Flexible Callbacks**: Pass custom functions for preprocessing, reward calculation, and state updates
- **Windows Integration**: Built-in Windows API utilities for screenshot capture and input control
- **DQN Implementation**: Example Deep Q-Network implementation (easily extensible to other algorithms)
- **Easy Customization**: Designed for quick adaptation to different games

## Project Structure

```
RL-based-AI-env/
├── src/
│   ├── core/                    # Core RL abstractions
│   │   ├── base_model.py        # Abstract base class for RL models
│   │   ├── base_environment.py  # Abstract base class for environments
│   │   └── replay_buffer.py     # Experience replay buffer
│   ├── models/                  # Model implementations
│   │   └── dqn_model.py         # DQN implementation
│   ├── agent/                   # RL agent
│   │   └── rl_agent.py          # Training/evaluation logic
│   ├── environment/             # Environment implementations
│   │   └── game_env.py          # Concrete game environment
│   └── utils/                   # Utilities
│       └── windows_controller.py # Windows API integration
├── examples/                    # Example usage
│   ├── train_example.py         # Training script with callbacks
│   └── play_example.py          # Play/demo script
├── checkpoints/                 # Saved models (created automatically)
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd RL-based-AI-env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Customize Callback Functions

Edit `examples/train_example.py` and customize these functions for your game:

```python
def preprocess_screenshot(screenshot: np.ndarray) -> np.ndarray:
    """Convert raw screenshot to model input"""
    # Your preprocessing logic
    pass

def extract_game_info(processed_screenshot: np.ndarray) -> dict:
    """Extract game state information (OCR, template matching, etc.)"""
    # Your extraction logic
    pass

def calculate_reward(prev_info: dict, current_info: dict) -> float:
    """Calculate reward based on state changes"""
    # Your reward logic
    pass

def update_state(current_state: dict, new_info: dict, action) -> dict:
    """Update internal state representation"""
    # Your state update logic
    pass
```

### 2. Configure Your Game

Set up game-specific parameters:

```python
WINDOW_NAME = "YourGameName"  # Game window title
ACTION_SPACE = ['w', 'a', 's', 'd', 'space']  # Possible actions
```

### 3. Train the Agent

```bash
python examples/train_example.py
```

### 4. Test the Trained Model

```bash
python examples/play_example.py
```

## Customization Guide

### Adding a New RL Algorithm

1. Create a new file in `src/models/` (e.g., `ppo_model.py`)
2. Inherit from `BaseModel` and implement required methods:
   - `predict()`: Action prediction
   - `train_step()`: Training step
   - `get_action()`: Action selection with exploration
   - `save()` / `load()`: Model persistence

Example:
```python
from src.core import BaseModel

class PPOModel(BaseModel):
    def predict(self, state):
        # Implementation
        pass

    def train_step(self, batch):
        # Implementation
        pass
```

### Customizing the Environment

Modify `src/environment/game_env.py` to:
- Change screenshot capture region
- Implement game reset logic
- Add complex action support (mouse movements, combinations)
- Customize episode termination conditions

### Advanced Features

#### Frame Stacking
For better temporal information, stack multiple frames:

```python
def preprocess_screenshot(screenshot: np.ndarray) -> np.ndarray:
    # Store in a frame buffer
    # Return stacked frames
    pass
```

#### Reward Shaping
Design sophisticated reward functions:

```python
def calculate_reward(prev_info, current_info):
    reward = 0
    # Distance-based rewards
    # Time-based bonuses
    # Sparse rewards for objectives
    return reward
```

## Tips and Best Practices

### 1. Start Simple
- Begin with simple preprocessing (grayscale, resize)
- Use basic reward signals first
- Test with short training episodes

### 2. Debug Callbacks
- Print extracted info to verify detection works
- Visualize preprocessed screenshots
- Log rewards to ensure they're reasonable

### 3. Hyperparameter Tuning
Key parameters to adjust:
- `learning_rate`: How fast the model learns
- `epsilon_decay`: Exploration vs exploitation
- `batch_size`: Training stability
- `gamma`: Future reward importance

### 4. Performance Optimization
- Use GPU for training (`torch.cuda`)
- Reduce screenshot resolution if slow
- Increase `frame_skip` for faster games
- Use efficient image processing (OpenCV)

### 5. Monitoring Training
- Watch episode rewards trend
- Check if agent explores enough
- Monitor loss values
- Save checkpoints frequently

## Common Issues

### Window Not Found
- Ensure game is running
- Check window name matches exactly
- Try using `win32gui.EnumWindows()` to list all windows

### Poor Performance
- Check if rewards are too sparse
- Verify state preprocessing is correct
- Ensure action space matches game controls
- Try different hyperparameters

### Training Not Improving
- Increase exploration (higher epsilon)
- Check reward function validity
- Verify state contains enough information
- Try larger replay buffer

## Future Enhancements

Potential additions for your project:

1. **GUI Dashboard**:
   - Real-time training metrics
   - Live screenshot display
   - Manual testing interface
   - Hyperparameter adjustment

2. **Multi-game Support**:
   - Game profiles/configs
   - Auto-detection of game type
   - Shared pretrained models

3. **Advanced Algorithms**:
   - PPO (Proximal Policy Optimization)
   - A3C (Asynchronous Actor-Critic)
   - SAC (Soft Actor-Critic)

4. **Data Collection**:
   - Record gameplay sessions
   - Human demonstration learning
   - Offline RL support

5. **Cloud Training**:
   - Distributed training
   - Remote monitoring
   - Model sharing

## Contributing

Feel free to extend and customize this framework for your needs!

## License

MIT License - feel free to use for personal projects

## Acknowledgments

Built with:
- PyTorch for deep learning
- pywin32 for Windows API
- OpenCV for image processing
