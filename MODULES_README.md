# Pac-Man RL Agent - Modular OOP Structure

This directory contains a modular, object-oriented implementation of a Deep Q-Learning agent for Pac-Man using the APEX (Asynchronous Prioritized Experience Replay) architecture.

## 📁 File Structure

```
PyCharmMiscProject/
├── pacman_reward_wrapper.py    # Environment wrapper for reward shaping
├── dqn_networks.py              # Neural network architectures (DQN, DuelingDQN)
├── rl_agent.py                  # Complete RL agent with prioritized replay
├── train_single.py              # Single-process DQN training script
├── evaluate_agent.py            # Model evaluation and testing utilities
├── play_agent.py                # Simple playback utility
├── apex_actor.py                # [DISTRIBUTED] Actor process for parallel collection
├── apex_learner.py              # [DISTRIBUTED] Learner process for training
├── train_apex.py                # [DISTRIBUTED] Multi-process training script
└── modules_readme.md            # This file
```

**Note:** Files marked [DISTRIBUTED] are for advanced APEX multi-process training.

---

## 📋 Quick Reference Table

| File | Class/Function | Key Methods | Return Values |
|------|----------------|-------------|---------------|
| `pacman_reward_wrapper.py` | `PacmanRewardWrapper` | `reset()`, `step()` | (obs, info), (obs, reward, term, trunc, info) |
| `dqn_networks.py` | `DQN`, `DuelingDQN` | `forward()` | Q-values tensor (batch, actions) |
| `rl_agent.py` | `RlAgent` | `act()`, `cache()`, `recall()`, `learn()` | int, None, tuple(7 tensors), (Q-val, loss) or (None, None) |
| `train_single.py` | - | `train_single_process()` | (episode_rewards, episode_lengths) |
| `evaluate_agent.py` | - | `evaluate_agent()`, `compare_models()`, `watch_agent_play()` | metrics dict, results dict, metrics dict |
| `play_agent.py` | `PlayAgent` | `play()` | None |

---

## 🎯 Module Descriptions

### 1. `pacman_reward_wrapper.py`
**Class:** `PacmanRewardWrapper(gym.Wrapper)`

Environment wrapper that provides balanced reward shaping for better training signals.

**Purpose:** Modifies Pac-Man's reward structure to make learning more effective by making dots valuable and death penalties recoverable.

#### Methods:

**`__init__(self, env)`**
- **Args:** `env` (gymnasium.Env) - Environment to wrap
- **Operations:** Initializes wrapper, sets lives tracking to 0
- **Returns:** None

**`reset(self, **kwargs)`**
- **Args:** `**kwargs` - Additional arguments passed to environment reset
- **Operations:**
  - Resets the wrapped environment
  - Initializes lives from info dict (defaults to 3)
  - Resets step counter to 0
- **Returns:** `tuple(observation, info)`

**`step(self, action)`**
- **Args:** `action` (int) - Action to take in environment
- **Operations:**
  - Executes action in wrapped environment
  - Applies reward shaping:
    - Positive rewards (dots/ghosts): multiply by 0.5 (dot=5, pellet=25, ghost=100-800)
    - Zero rewards (no action): apply -0.001 penalty
    - Life lost: apply -5.0 penalty
  - Tracks lives and steps
- **Returns:** `tuple(observation, shaped_reward, terminated, truncated, info)`

**Key Reward Balancing:**
- Eating 1 dot: +5.0 (recovers from death)
- Death penalty: -5.0 (recoverable)
- Idle time penalty: -0.001 per step

---

### 2. `dqn_networks.py`
**Classes:** `DQN(nn.Module)`, `DuelingDQN(nn.Module)`

Neural network architectures for Q-value estimation in Deep Q-Learning.

#### Class: `DQN`
Standard Deep Q-Network with convolutional feature extraction.

**`__init__(self, input_shape, num_actions)`**
- **Args:**
  - `input_shape` (tuple or int) - Input dimensions (channels, height, width) or channels only
  - `num_actions` (int) - Number of possible actions
- **Operations:**
  - Creates 3 convolutional layers (32, 64, 64 filters)
  - Creates 2 fully connected layers (512, num_actions)
- **Returns:** None
- **Architecture:** Conv(8,4) → Conv(4,2) → Conv(3,1) → FC(3136→512) → FC(512→actions)

**`forward(self, x)`**
- **Args:** `x` (tensor) - Input state tensor, shape (batch, channels, height, width)
- **Operations:**
  - Passes through 3 conv layers with ReLU activation
  - Flattens to (batch, 3136)
  - Passes through 2 FC layers
- **Returns:** `tensor` - Q-values for each action, shape (batch, num_actions)

#### Class: `DuelingDQN`
Dueling architecture that separates value function V(s) and advantage function A(s,a).

**Formula:** Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

**`__init__(self, input_shape, num_actions)`**
- **Args:** Same as DQN
- **Operations:**
  - Creates shared convolutional feature extractor (3 layers)
  - Creates value stream: FC(3136→512) → FC(512→1)
  - Creates advantage stream: FC(3136→512) → FC(512→num_actions)
- **Returns:** None

**`forward(self, x)`**
- **Args:** `x` (tensor) - Input state tensor, shape (batch, channels, height, width)
- **Operations:**
  - Shared feature extraction through 3 conv layers
  - Flattens to (batch, 3136)
  - Value stream: computes V(s), shape (batch, 1)
  - Advantage stream: computes A(s,a), shape (batch, num_actions)
  - Combines using dueling formula to ensure identifiability
- **Returns:** `tensor` - Q-values for each action, shape (batch, num_actions)

---

### 3. `rl_agent.py`
**Class:** `RlAgent`

Complete Deep Q-Learning agent with prioritized experience replay and Double DQN.

**Purpose:** Implements full DQN training pipeline with state-of-the-art improvements.

#### Methods:

**`__init__(self, input_dim, output_dim, save_dir=None, use_dueling=True)`**
- **Args:**
  - `input_dim` (tuple) - Input dimensions (channels, height, width)
  - `output_dim` (int) - Number of actions
  - `save_dir` (str, optional) - Directory for saving models
  - `use_dueling` (bool) - Whether to use Dueling DQN (default: True)
- **Operations:**
  - Initializes policy and target networks (DuelingDQN or DQN)
  - Sets device (CUDA > MPS > CPU)
  - Creates Adam optimizer (lr=2.5e-4)
  - Initializes PrioritizedReplayBuffer (capacity=100k, alpha=0.6, beta=0.4)
  - Sets hyperparameters: batch_size=64, gamma=0.99, exploration=1.0→0.01
- **Returns:** None

**`act(self, state)`**
- **Args:** `state` (np.array) - Current observation from environment
- **Operations:**
  - Epsilon-greedy action selection
  - If exploring: random action
  - If exploiting: argmax Q-value from policy network
  - Decays exploration_rate by 0.99999
  - Increments curr_step counter
- **Returns:** `int` - Action index to take

**`cache(self, states, actions, next_states, rewards, done)`**
- **Args:**
  - `states` (np.array) - Current state
  - `actions` (int) - Action taken
  - `next_states` (np.array) - Next state
  - `rewards` (float) - Reward received
  - `done` (bool) - Episode termination flag
- **Operations:**
  - Converts inputs to tensors on device
  - Creates TensorDict with experience data
  - Adds to PrioritizedReplayBuffer
- **Returns:** None

**`recall(self)`**
- **Args:** None
- **Operations:**
  - Samples batch_size (64) experiences from prioritized replay buffer
  - Retrieves indices and importance sampling weights
  - Extracts states, actions, next_states, rewards, dones from TensorDict
- **Returns:** `tuple` - (states, actions, next_states, rewards, dones, indices, weights)
  - All tensors on device with batch_size=64

**`learn(self)`**
- **Args:** None
- **Operations:**
  - **Step 1:** Syncs target network every sync_every (10k) steps
  - **Step 2:** Checks burnin period (5k steps) before learning
  - **Step 3:** Only learns every learn_every (4) steps
  - **Step 4:** Anneals beta from 0.4 → 1.0 over 100k steps
  - **Step 5:** Samples batch from memory via recall()
  - **Step 6:** Computes TD estimates from policy network
  - **Step 7:** Computes TD targets using Double DQN:
    - Policy net selects best action
    - Target net evaluates that action
    - Bellman equation: r + γ * Q_target(s', a')
  - **Step 8:** Computes weighted Smooth L1 loss with importance sampling
  - **Step 9:** Backpropagates with gradient clipping (max_norm=10.0)
  - **Step 10:** Updates priorities in buffer based on TD-errors
- **Returns:** `tuple(float, float)` or `tuple(None, None)`
  - (mean Q-value, loss) if learning occurred
  - (None, None) if skipped (burnin/learn_every)

**`save_model(self, filepath, episode=None, metadata=None)`**
- **Args:**
  - `filepath` (str) - Path to save checkpoint
  - `episode` (int, optional) - Current episode number
  - `metadata` (dict, optional) - Additional metadata
- **Operations:**
  - Saves policy_net, target_net, optimizer state dicts
  - Saves exploration_rate, curr_step, architecture type
  - Optionally saves episode and metadata
- **Returns:** None

**`load_model(self, filepath, load_optimizer=True)`**
- **Args:**
  - `filepath` (str) - Path to checkpoint
  - `load_optimizer` (bool) - Whether to load optimizer state
- **Operations:**
  - Loads checkpoint to device
  - Restores policy_net and target_net weights
  - Optionally restores optimizer, exploration_rate, curr_step
- **Returns:** `dict` - Checkpoint dictionary with metadata

---

### 4. `train_single.py`
**Function:** `train_single_process()`

Single-process DQN training script for Pac-Man (non-distributed).

**Purpose:** Synchronous training where agent collects experience, stores it, and learns immediately.

#### Functions:

**`train_single_process(episodes=1000, frameskip=4, save_dir="saved_models", save_every=100, print_every=250)`**
- **Args:**
  - `episodes` (int) - Number of episodes to train
  - `frameskip` (int) - Frame skip for environment
  - `save_dir` (str) - Directory for saving checkpoints
  - `save_every` (int) - Save model every N episodes
  - `print_every` (int) - Print progress every N episodes
- **Operations:**
  - **Setup:** Creates Pac-Man environment with preprocessing wrappers (Resize→84x84, Grayscale, FrameStack→4, PacmanReward)
  - **Agent creation:** Initializes RlAgent with Dueling DQN
  - **Training loop (per episode):**
    - Resets environment
    - **Step loop:**
      - Agent selects action via `act()`
      - Environment steps forward
      - Experience stored via `cache()`
      - Agent learns via `learn()`
      - Tracks Q-values and losses
    - Tracks episode reward, length
  - **Logging:** Prints avg reward (last 100), Q-values, loss, exploration rate
  - **Saving:** Saves checkpoints with metadata (total_steps, avg_reward_100)
- **Returns:** `tuple(list, list)` - (episode_rewards, episode_lengths)

**`main()`**
- **Args:** None
- **Operations:** Entry point that calls `train_single_process()` with default or custom parameters
- **Returns:** None

**Key Training Flow:**
```
For each episode:
  Reset environment
  While not done:
    1. act(state) → action
    2. env.step(action) → next_state, reward
    3. cache(experience) → store in replay buffer
    4. learn() → sample batch & update network
```

---

### 5. `evaluate_agent.py`
**Functions:** `evaluate_agent()`, `compare_models()`, `watch_agent_play()`

Evaluation and testing utilities for trained agents.

**Purpose:** Load trained models and measure performance metrics without training.

#### Functions:

**`evaluate_agent(model_path, num_episodes=10, render=False, exploration_rate=0.01, frameskip=4)`**
- **Args:**
  - `model_path` (str) - Path to checkpoint (.pth file)
  - `num_episodes` (int) - Number of test episodes
  - `render` (bool) - Whether to display game window
  - `exploration_rate` (float) - Epsilon for evaluation (low for greedy)
  - `frameskip` (int) - Frame skip
- **Operations:**
  - Creates environment with same preprocessing as training
  - Creates RlAgent and loads checkpoint weights
  - Sets policy_net to eval mode
  - Runs num_episodes episodes:
    - Tracks episode reward, length, raw Atari score
    - Uses mostly greedy policy (low epsilon)
  - Computes statistics: mean, std, min, max rewards
- **Returns:** `dict` - Metrics dictionary:
  ```python
  {
    'avg_reward': float,
    'std_reward': float,
    'min_reward': float,
    'max_reward': float,
    'avg_length': float,
    'avg_score': float,
    'max_score': float
  }
  ```

**`compare_models(model_paths, num_episodes=10)`**
- **Args:**
  - `model_paths` (list[str]) - List of checkpoint paths to compare
  - `num_episodes` (int) - Episodes to evaluate each model
- **Operations:**
  - Calls `evaluate_agent()` for each model
  - Collects all metrics
  - Prints comparison table with avg reward and score
- **Returns:** `dict` - Dictionary mapping model_path → metrics

**`watch_agent_play(model_path, num_episodes=5, exploration_rate=0.0, frameskip=4, slow_mode=False)`**
- **Args:**
  - `model_path` (str) - Path to checkpoint
  - `num_episodes` (int) - Episodes to watch
  - `exploration_rate` (float) - Epsilon (0.0 for fully greedy)
  - `frameskip` (int) - Frame skip
  - `slow_mode` (bool) - Add delay for viewing
- **Operations:**
  - Calls `evaluate_agent()` with render=True
  - Opens game window to watch agent play
  - Optionally adds time.sleep() for slower playback
- **Returns:** `dict` - Same metrics as evaluate_agent()

**`main()`**
- **Args:** Command-line arguments via argparse
- **Operations:** Parses args and calls appropriate evaluation function
- **Returns:** None

---

### 6. `play_agent.py`
**Class:** `PlayAgent`

Simple playback utility to watch trained agent play with visualization.

**Purpose:** Lightweight script to load weights and render gameplay.

#### Methods:

**`__init__(self)`**
- **Args:** None
- **Operations:**
  - Creates Pac-Man environment with render_mode='human'
  - Applies preprocessing wrappers (Grayscale, Resize→84x84, FrameStack→4, PacmanReward)
  - Initializes DuelingDQN network
  - Sets device (MPS or CPU)
  - Moves network to device
- **Returns:** None

**`play(self, checkpoint_path, num_episodes=5)`**
- **Args:**
  - `checkpoint_path` (str) - Path to .pth checkpoint file
  - `num_episodes` (int) - Number of episodes to play
- **Operations:**
  - Loads checkpoint using torch.load()
  - Handles multiple checkpoint formats:
    - 'policy_net_state_dict' (APEX format)
    - 'model_state_dict' (standard format)
    - Direct state_dict
  - Sets network to eval mode
  - **Game loop (per episode):**
    - Resets environment
    - Selects actions greedily (argmax Q-values)
    - Steps environment and renders
    - Tracks total reward and steps
  - Prints per-episode stats and summary stats
  - Closes environment when done
- **Returns:** None

**Example Usage:**
```python
player = PlayAgent()
player.play(checkpoint_path='saved_models/apex_final.pth', num_episodes=3)
```

---

## 🚀 Quick Start

### Option 1: Single-Process Training (Recommended for Beginners)

Train a DQN agent with a simple, synchronous approach:

```bash
python train_single.py
```

This runs 1000 episodes of training and saves checkpoints to `saved_models/`.

**Customize parameters:**
```python
from train_single import train_single_process

train_single_process(
    episodes=1000,
    frameskip=4,
    save_dir="saved_models",
    save_every=100,
    print_every=10
)
```

### Option 2: Evaluate Trained Agent

After training, evaluate performance:

```bash
python evaluate_agent.py --model saved_models/dqn_final.pth --episodes 10
```

Or with rendering to watch the agent play:

```bash
python evaluate_agent.py --model saved_models/dqn_final.pth --episodes 5 --render
```

**In Python:**
```python
from evaluate_agent import evaluate_agent, watch_agent_play

# Evaluate without rendering
metrics = evaluate_agent(
    model_path="saved_models/dqn_final.pth",
    num_episodes=10,
    render=False
)

# Watch the agent play
watch_agent_play(
    model_path="saved_models/dqn_final.pth",
    num_episodes=3,
    exploration_rate=0.0  # Fully greedy
)
```

### Option 3: Simple Playback

Quick way to watch a trained agent:

```python
from play_agent import PlayAgent

player = PlayAgent()
player.play(checkpoint_path='saved_models/dqn_final.pth', num_episodes=3)
```

### Option 4: Custom Training Loop

Build your own training pipeline using individual modules:

```python
from rl_agent import RlAgent
from pacman_reward_wrapper import PacmanRewardWrapper
import gymnasium as gym
import ale_py
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation

# Setup environment
gym.register_envs(ale_py)
env = gym.make('ALE/Pacman-v5', frameskip=4)
env = ResizeObservation(env, (84, 84))
env = GrayscaleObservation(env)
env = FrameStackObservation(env, 4)
env = PacmanRewardWrapper(env)

# Create agent
agent = RlAgent(
    input_dim=(4, 84, 84),
    output_dim=env.action_space.n,
    use_dueling=True
)

# Training loop
for episode in range(1000):
    state, _ = env.reset()

    while True:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.cache(state, action, next_state, reward, done)
        q_val, loss = agent.learn()

        if done:
            break

        state = next_state
```

### Option 5: Distributed APEX Training (Advanced)

For multi-process parallel training:

```bash
python train_apex.py
```

**Note:** APEX is incompatible with MPS (Apple Silicon GPU). Use single-process training for GPU support on Mac.

---

## 🔄 Batch Training Flow Explained

Understanding how batch training works in this DQN implementation:

### Experience Collection (Per-Step)
Each step in the environment generates **one experience** (transition):
```python
# In train_single.py
action = agent.act(state)                    # Select action
next_state, reward, done = env.step(action)  # Execute action
agent.cache(state, action, next_state, reward, done)  # Store immediately
```

**Key Point:** Experiences are stored **one at a time**, not after episode completion.

### Batch Sampling
When `agent.learn()` is called:
```python
# In rl_agent.py - recall()
batch = memory.sample(batch_size=64)  # Sample 64 random experiences
# Returns: states, actions, next_states, rewards, dones, indices, weights
```

The prioritized replay buffer samples based on TD-error priorities:
- High TD-error experiences → sampled more frequently
- Low TD-error experiences → sampled less frequently

### Batch Processing
The entire batch (64 experiences) is processed in parallel:

```python
# In rl_agent.py - learn()
# Step 1: Compute Q-values for all 64 states simultaneously
td_est = policy_net(states).gather(1, actions)  # Shape: (64, 1)

# Step 2: Compute targets using Double DQN for all 64 next_states
best_actions = policy_net(next_states).argmax(1)
next_q = target_net(next_states).gather(1, best_actions)
td_tgt = rewards + gamma * next_q * (1 - dones)  # Shape: (64, 1)

# Step 3: Compute weighted loss across all 64 experiences
loss = (smooth_l1_loss(td_est, td_tgt) * weights).mean()

# Step 4: Backpropagate on batch
loss.backward()
optimizer.step()
```

### Training Schedule

| Parameter | Value | Description |
|-----------|-------|-------------|
| `burnin` | 5,000 steps | Wait before starting training |
| `learn_every` | 4 steps | Learn every 4 environment steps |
| `batch_size` | 64 | Experiences per training update |
| `sync_every` | 10,000 steps | Update target network frequency |

**Example Timeline:**
- Steps 1-5000: Collect experiences only (burnin)
- Steps 5004, 5008, 5012...: Learn from batch of 64
- Step 10,000: Sync target network with policy network

### Why Batch Training?

1. **Efficiency:** GPU processes 64 states in parallel (much faster than sequential)
2. **Stability:** Averaging gradient across batch reduces variance
3. **Sample Efficiency:** Each experience can be used multiple times via replay buffer
4. **Breaking Correlation:** Random sampling breaks temporal correlation between consecutive states

### Experience Lifecycle

```
Step 1: Collect → cache() → PrioritizedReplayBuffer (capacity: 100k)
                              ↓
Step 2: Sample  → recall() → Random batch of 64 with priorities
                              ↓
Step 3: Learn   → learn() → Compute loss, backprop, update priorities
                              ↓
Step 4: Reuse   → recall() → Same experience can be sampled again later
```

An experience can be used **multiple times** during training until it's evicted from the buffer (FIFO when full).

---

## ⚠️ Important Notes

### Multiprocessing Requirements

1. **Always use `if __name__ == '__main__':` guard:**
   ```python
   if __name__ == '__main__':
       train_apex(...)
   ```

2. **Set spawn method on Mac/Windows:**
   ```python
   import multiprocessing as mp
   mp.set_start_method('spawn', force=True)
   ```

3. **Run from Python scripts, not Jupyter notebooks:**
   - Multiprocessing doesn't work well in Jupyter
   - Classes must be importable from modules
   - Use `.py` files instead of `.ipynb`

### MPS (Apple Silicon) Compatibility

APEX uses multiprocessing, which is **incompatible with MPS**. The code automatically falls back to CPU when MPS is detected.

For GPU training on Apple Silicon, use the single-process `RlAgent` directly instead of APEX.

---

## 📊 Model Checkpoints

Training saves models in the `saved_models/` directory:

- **`apex_final.pth`** - Final trained model
- **`apex_checkpoint_N.pth`** - Checkpoints every N experiences

### Loading a Trained Model

```python
from rl_agent import RlAgent
import torch

agent = RlAgent(input_dim=(4, 84, 84), output_dim=9)
checkpoint = agent.load_model('saved_models/apex_final.pth')

print(f"Loaded model from episode {checkpoint.get('episode', 'unknown')}")
```

---

## 🧪 Testing

Start with small parameters to verify everything works:

```python
train_apex(
    num_actors=2,          # Just 2 actors
    episodes_per_actor=10, # 10 episodes each
    checkpoint_every=100   # Checkpoint every 100 experiences
)
```

Expected output:
```
🚀 APEX TRAINING - Asynchronous Prioritized Experience Replay
================================================================================
[Learner] Started on cpu
[Actor 0] Started (ε: 1.000 -> 0.010)
[Actor 1] Started (ε: 0.900 -> 0.030)
...
✅ APEX TRAINING COMPLETE!
```

---

## 🏗️ Architecture Overview

```
Main Process
├── Creates shared queues
├── Spawns Learner Process
│   ├── Creates RlAgent (with PrioritizedReplayBuffer)
│   ├── Receives experiences from queue
│   ├── Trains network
│   └── Broadcasts weights to actors
│
└── Spawns Actor Processes (parallel)
    ├── Creates environment
    ├── Creates policy network
    ├── Collects experiences
    ├── Sends experiences to queue
    └── Syncs weights from learner
```

---

## 📚 Dependencies

```
torch
torchrl
tensordict
gymnasium
ale-py
numpy
```

Install:
```bash
pip install torch torchrl tensordict gymnasium ale-py numpy
```

---

## 🐛 Troubleshooting

### Error: "Can't get attribute 'APEXLearner' on <module '__main__'>"

**Solution:** You're trying to run multiprocessing from a Jupyter notebook. Use a `.py` file instead.

### Error: "RuntimeError: Cannot share a storage of type LazyMemmapStorage"

**Solution:** Already fixed! The `RlAgent` is created inside `run()` method, not in `__init__()`.

### Actors start but don't collect experiences

**Solution:** Make sure weight queue is working. Increase learner initialization time:
```python
learner_process.start()
time.sleep(5)  # Give more time
```

### Memory grows continuously

**Solution:** Increase `learn_every` or reduce `num_actors`:
```python
agent.learn_every = 8  # Train less frequently
```

---

## 📖 Further Reading

- [APEX Paper](https://arxiv.org/abs/1803.00933) - Original algorithm
- [TorchRL Documentation](https://pytorch.org/rl/) - Replay buffers
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581) - Network architecture

---

## ✅ Summary

This modular structure:
- ✅ Fixes multiprocessing errors (agent created in subprocess)
- ✅ Uses proper OOP design with separate, importable modules
- ✅ Includes complete documentation and examples
- ✅ Works with Python scripts (not just notebooks)
- ✅ Supports both APEX and single-process training
- ✅ Includes reward shaping for better training
- ✅ Uses Dueling DQN and prioritized replay for performance

Happy training! 🚀🎮
