# Dr. Pac-Man: Deep Q-Learning Agent

## Overview
This repository implements a Deep Q-Network (DQN) agent for playing Atari Pac-Man using PyTorch. Features include Dueling DQN architecture, Prioritized Experience Replay, custom reward shaping, and automatic checkpoint discovery for seamless training resumption.

## Key Features
- **Dueling DQN Architecture**: Separate value and advantage streams for better Q-value estimation
- **Prioritized Experience Replay**: Learn from important experiences more frequently
- **Custom Reward Shaping**: Balanced rewards for pellets, ghosts, and survival
- **Automatic Checkpoint Discovery**: Seamlessly resume training without manual configuration
- **Apple Silicon Support**: Full MPS (Metal Performance Shaders) support for M1/M2/M3 Macs
- **Comprehensive Evaluation Tools**: Compare models, watch gameplay, analyze metrics

## Running Training
The training script `train_single.py` trains the agent using single-process DQN with automatic checkpoint discovery.

```bash
python train_single.py
```

### Training Modes
```python
# Auto-resume (default - finds best checkpoint automatically)
train_single_process(episodes=1000)

# Force fresh start (ignore existing checkpoints)
train_single_process(episodes=1000, checkpoint=False)

# Load specific checkpoint
train_single_process(episodes=1000, checkpoint="saved_models/dqn_episode_500.pth")
```

### Configuration Parameters
Edit `train_single.py` to customize:
| Parameter | Description | Default |
|----------|-------------|---------|
| `episodes` | Number of episodes to train | 1000 |
| `frameskip` | Frames to skip per action | 4 |
| `save_dir` | Checkpoint directory | saved_models |
| `save_every` | Save checkpoint every N episodes | 100 |
| `checkpoint` | Checkpoint path or mode | None (auto) |

#### Example: Quick test run
```python
# In train_single.py main() function
train_single_process(
    episodes=10,
    save_every=5,
    print_every=1
)
```

## Running Evaluation
After training, evaluate the model with `evaluate_agent.py`.

```bash
# Evaluate final model (headless)
python evaluate_agent.py --model saved_models/dqn_final.pth --episodes 10

# Watch the agent play with rendering
python evaluate_agent.py --model saved_models/dqn_final.pth --episodes 5 --render
```

### Additional options
- `--render`: Render the game to watch the agent (requires a display).
- `--compare MODEL1 MODEL2`: Compare multiple checkpoints.
- `--epsilon VALUE`: Set exploration rate (default 0.01).

#### Example: Compare checkpoints
```bash
python evaluate_agent.py --model saved_models/dqn_episode_500.pth --compare saved_models/dqn_episode_1000.pth saved_models/dqn_final.pth
```

#### Quick playback utility
```python
# Simple script to watch agent play
python -c "from play_agent import PlayAgent; PlayAgent().play('saved_models/dqn_final.pth', 3)"
```

## Project Structure
```
PyCharmMiscProject/
├── train_single.py              # Main training script with auto-discovery
├── rl_agent.py                  # DQN agent with prioritized replay
├── dqn_networks.py              # DQN and Dueling DQN architectures
├── pacman_reward_wrapper.py     # Custom reward shaping
├── evaluate_agent.py            # Model evaluation and comparison
├── play_agent.py                # Simple playback utility
├── saved_models/                # Checkpoint directory (git-ignored)
│   ├── dqn_final.pth           # Final trained model
│   └── dqn_episode_*.pth       # Periodic checkpoints
├── CLAUDE.md                    # Project instructions for Claude Code
└── MODULES_README.md            # Detailed module documentation
```

## Tips & Troubleshooting
- **Apple Silicon Macs**: Training automatically uses MPS (Metal) for acceleration
- **CUDA GPUs**: Automatically detected and used if available
- **Resume Training**: Just run `python train_single.py` - it auto-resumes from best checkpoint
- **Start Fresh**: Use `checkpoint=False` in code or delete checkpoints from `saved_models/`
- **Checkpoint Files**: All `.pth` files are git-ignored to keep repository lightweight

