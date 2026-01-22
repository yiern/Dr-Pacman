# AGENTS.md

This file provides guidance for AI agents working in this repository.

## Project Overview

Deep Q-Learning (DQN) agent for Atari Pac-Man using PyTorch. Implements Dueling DQN architecture, Prioritized Experience Replay, and custom reward shaping.

## Build/Lint/Test Commands

### Virtual Environment
```bash
source .venv/bin/activate
.venv/bin/pip install torch gymnasium ale-py numpy torchrl tensordict
```

### Training Commands
```bash
# Full training (1000 episodes)
python train_single.py

# Quick test run (3 episodes, frequent saves)
python -c "
from train_single import train_single_process
train_single_process(
    episodes=3,
    save_every=1,
    print_every=1
)
"
```

### Evaluation Commands
```bash
# Evaluate model (headless)
python evaluate_agent.py --model saved_models/dqn_final.pth --episodes 10

# Watch agent play
python evaluate_agent.py --model saved_models/dqn_final.pth --episodes 5 --render
```

### Single Test Pattern
There is no formal test suite. Test a single component by running training with minimal episodes:
```python
from train_single import train_single_process
train_single_process(episodes=3, save_every=1, print_every=1)
```

## Code Style Guidelines

### Imports
Organize in three groups separated by blank lines:
1. Standard library (`os`, `re`, `collections`, `argparse`)
2. Third-party packages (`torch`, `gymnasium`, `numpy`)
3. Local modules (relative imports, alphabetically sorted)

```python
import os
import re
from collections import namedtuple

import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation

from dqn_networks import DQN, DuelingDQN
from pacman_reward_wrapper import PacmanRewardWrapper
```

### Formatting
- Line length: 100 characters maximum
- 4 spaces for indentation (no tabs)
- Two blank lines between top-level class/function definitions
- One blank line between method definitions in a class
- No trailing whitespace

### Type Hints
- Use type hints for function parameters and return values where beneficial
- Simple types: `int`, `str`, `bool`, `float`
- Complex types: `List[Tuple[str, int]]`, `Dict[str, Any]`
- Omit type hints for `self`/`cls`

```python
def evaluate_agent(
    model_path: str,
    num_episodes: int = 10,
    render: bool = False
) -> Optional[Dict[str, float]]:
```

### Naming Conventions
| Type | Convention | Examples |
|------|------------|----------|
| Classes | PascalCase | `RlAgent`, `DuelingDQN`, `PacmanRewardWrapper` |
| Functions | snake_case | `train_single_process`, `find_latest_checkpoint` |
| Variables | snake_case | `exploration_rate`, `episode_rewards`, `curr_step` |
| Constants | UPPER_SNAKE_CASE | `MAX_BUFFER_SIZE = 100000` |
| Private methods | prefix with `_` | `_compute_td_loss()`, `_sync_target()` |

### Docstrings
Use Google-style docstrings for all public classes and functions.

### Error Handling
- Use specific exception types (`FileNotFoundError`, `ValueError`, `KeyError`)
- Provide informative error messages
- Use try/except for expected failure cases

### File Organization
- One module per file (max ~300 lines)
- Order: module docstring → imports → constants → classes → functions → `if __name__ == '__main__':`

### ML-Specific Conventions
- Device priority: CUDA > MPS > CPU
- Use `with torch.no_grad():` for inference
- Model checkpoint format includes `model_state_dict`, `target_state_dict`, `optimizer_state_dict`, `exploration_rate`, `curr_step`, `architecture`
- Set `model.eval()` for evaluation, `model.train()` for training

### Environment Wrapper Order
Apply wrappers in this exact order:
1. `ResizeObservation(84, 84)`
2. `GrayscaleObservation()`
3. `FrameStackObservation(4)`
4. `PacmanRewardWrapper()`

## Key Files

| File | Purpose |
|------|---------|
| `train_single.py` | Main training script |
| `rl_agent.py` | DQN agent implementation |
| `dqn_networks.py` | Neural network architectures |
| `pacman_reward_wrapper.py` | Reward shaping wrapper |
| `evaluate_agent.py` | Model evaluation utilities |
| `play_agent.py` | Simple playback utility |
| `saved_models/` | Checkpoint directory (git-ignored) |
