# Pac-Man APEX Project

## Overview
This repository implements Asynchronous Prioritized Experience Replay (APEX) for training and evaluating a Pac-Man agent in the Atari gym environment using PyTorch.

## Prerequisites
- Python 3.9+
- PyTorch (CPU or GPU)
- Git
- System dependencies for ALE (e.g., libsdl2-dev on Linux, Homebrew packages on macOS)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yiern/Dr-Pacman.git
   cd Dr-Pacman
   ```
2. Activate the existing virtual environment:
   ```bash
   source .venv/bin/activate  # macOS/Linux
   .\\.venv\\Scripts\\activate  # Windows
   ```
3. Install required Python packages using the environment's pip:
   ```bash
   .venv/bin/pip install torch gymnasium ale-py jsonwebtoken
   ```

## Running Training
The training script `train_apex.py` trains the agent using multiple parallel actors.

```bash
python train_apex.py
```

### Common arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--num_actors` | Number of parallel actors | 4 |
| `--episodes_per_actor` | Episodes per actor | 500 |
| `--frameskip` | Number of frames to skip per action | 4 |
| `--save_dir` | Directory to save checkpoints | saved_models |
| `--checkpoint_every` | Save checkpoint every N experiences | 10000 |

#### Example: Quick test run
```bash
python train_apex.py --num_actors 2 --episodes_per_actor 10 --save_dir saved_models_test
```

## Running Evaluation
After training, evaluate the model with `evaluate_agent.py`.

```bash
python evaluate_agent.py --model saved_models/apex_final.pth --episodes 10
```

### Additional options
- `--render`: Render the game to watch the agent (requires a display).
- `--compare MODEL1 MODEL2`: Compare multiple checkpoints.
- `--epsilon VALUE`: Set exploration rate (default 0.01).

#### Example: Compare two checkpoints
```bash
python evaluate_agent.py --model saved_models/apex_checkpoint_10000.pth --compare saved_models/apex_checkpoint_50000.pth saved_models/apex_final.pth
```

## Tips & Troubleshooting
- On macOS you may see an MPS warning; training will fall back to CPU.
- Ensure you have run training before evaluation; the default model path expects `saved_models/apex_final.pth`.
- Install system libraries if you encounter ALE errors:
  - Ubuntu: `sudo apt-get install libsdl2-dev libsdl2-image-dev libopenblas-dev`
  - macOS (Homebrew): `brew install sdl2`
- Increase `--num_actors` or use a GPU to accelerate training.

## License
This project is licensed under the MIT License – see the `LICENSE` file for details.