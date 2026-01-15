# Pac-Man RL Agent - Modular OOP Structure

This directory contains a modular, object-oriented implementation of a Deep Q-Learning agent for Pac-Man using the APEX (Asynchronous Prioritized Experience Replay) architecture.

## 📁 File Structure

```
PyCharmMiscProject/
├── pacman_reward_wrapper.py    # Environment wrapper for reward shaping
├── dqn_networks.py              # Neural network architectures (DQN, DuelingDQN)
├── rl_agent.py                  # Complete RL agent with prioritized replay
├── apex_actor.py                # Actor process for parallel experience collection
├── apex_learner.py              # Learner process for centralized training
├── train_apex.py                # Main training script
└── MODULES_README.md            # This file
```

## 🎯 Module Descriptions

### 1. `pacman_reward_wrapper.py`
**Class:** `PacmanRewardWrapper`

Environment wrapper that provides balanced reward shaping:
- Makes dots more valuable (+5.0 instead of +1.0)
- Reduces death penalty to be recoverable (-5.0)
- Adds small time penalty to encourage action (-0.001)

**Usage:**
```python
from pacman_reward_wrapper import PacmanRewardWrapper
import gymnasium as gym

env = gym.make('ALE/Pacman-v5')
env = PacmanRewardWrapper(env)
```

---

### 2. `dqn_networks.py`
**Classes:** `DQN`, `DuelingDQN`

Neural network architectures for Q-learning:

- **DQN**: Standard Deep Q-Network with 3 conv layers + 2 FC layers
- **DuelingDQN**: Separates value and advantage streams for better performance

**Usage:**
```python
from dqn_networks import DuelingDQN

network = DuelingDQN(input_shape=(4, 84, 84), num_actions=9)
q_values = network(state_tensor)
```

---

### 3. `rl_agent.py`
**Class:** `RlAgent`

Complete Deep Q-Learning agent with:
- Epsilon-greedy exploration with decay
- Prioritized Experience Replay with importance sampling
- Double DQN for stable Q-value estimation
- Beta annealing for bias correction
- Gradient clipping and monitoring

**Usage:**
```python
from rl_agent import RlAgent

agent = RlAgent(
    input_dim=(4, 84, 84),
    output_dim=9,
    use_dueling=True
)

# Collect experience
action = agent.act(state)
agent.cache(state, action, next_state, reward, done)

# Train
q_value, loss = agent.learn()
```

---

### 4. `apex_actor.py`
**Class:** `APEXActor`

Actor process for parallel experience collection:
- Runs in separate process
- Collects experiences by playing the game
- Syncs weights from learner periodically
- Each actor has different exploration parameters

**Usage:**
```python
from apex_actor import APEXActor
from multiprocessing import Queue, Process

experience_queue = Queue()
weight_queue = Queue()

actor = APEXActor(actor_id=0, experience_queue, weight_queue)
process = Process(target=actor.run, args=(500,))  # 500 episodes
process.start()
```

---

### 5. `apex_learner.py`
**Class:** `APEXLearner`

Centralized learner for APEX architecture:
- Receives experiences from actors via queue
- Trains network using prioritized replay
- Broadcasts updated weights to actors
- Saves checkpoints periodically

**IMPORTANT:** Creates `RlAgent` inside `run()` method to avoid multiprocessing errors.

**Usage:**
```python
from apex_learner import APEXLearner
from multiprocessing import Queue, Process

experience_queue = Queue()
weight_queues = [Queue(), Queue()]  # One per actor

learner = APEXLearner(
    input_dim=(4, 84, 84),
    output_dim=9,
    experience_queue,
    weight_queues
)

process = Process(target=learner.run, args=(100000,))  # Process 100k experiences
process.start()
```

---

### 6. `train_apex.py`
**Function:** `train_apex()`

Complete training script that:
- Sets up environment and queues
- Creates and starts learner process
- Creates and starts actor processes
- Waits for training to complete
- Saves trained models

**Usage:**
```bash
python train_apex.py
```

Or customize:
```python
from train_apex import train_apex

train_apex(
    num_actors=4,
    episodes_per_actor=500,
    save_dir="saved_models"
)
```

---

## 🚀 Quick Start

### Option 1: Run the Training Script

```bash
python train_apex.py
```

This will train with default test parameters (2 actors, 10 episodes each).

### Option 2: Import in Your Own Script

```python
from train_apex import train_apex
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    train_apex(
        num_actors=4,
        episodes_per_actor=500,
        frameskip=4,
        save_dir="saved_models",
        checkpoint_every=10000
    )
```

### Option 3: Use Individual Classes

```python
from rl_agent import RlAgent
from pacman_reward_wrapper import PacmanRewardWrapper
import gymnasium as gym
import ale_py

# Setup environment
gym.register_envs(ale_py)
env = gym.make('ALE/Pacman-v5')
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
        next_state, reward, done, truncated, info = env.step(action)

        agent.cache(state, action, next_state, reward, done)
        q_val, loss = agent.learn()

        if done or truncated:
            break

        state = next_state
```

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
