"""
Reinforcement Learning Agent with Prioritized Experience Replay

This module implements a complete DQN agent with:
- Dueling DQN or standard DQN architecture
- Prioritized Experience Replay with importance sampling
- Double DQN for stable Q-value estimation
- Beta annealing for bias correction
- Gradient clipping and monitoring
"""

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

from torchrl.data import LazyMemmapStorage, PrioritizedReplayBuffer
from tensordict import TensorDict

from dqn_networks import DQN, DuelingDQN


# Named tuple for experiences (kept for compatibility)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class RlAgent:
    """
    Deep Q-Learning Agent with Prioritized Experience Replay.

    This agent implements a complete DQN training pipeline with:
    - Epsilon-greedy exploration
    - Prioritized replay buffer with importance sampling
    - Double DQN target updates
    - Optional Dueling DQN architecture
    """

    def __init__(self, input_dim, output_dim, save_dir=None, use_dueling=True):
        """
        Initialize the RL agent.

        Args:
            input_dim: Input dimensions (channels, height, width)
            output_dim: Number of actions
            save_dir: Directory to save models (optional)
            use_dueling: Whether to use Dueling DQN (default: True)
        """
        self.state_dim = input_dim
        self.action_dim = output_dim
        self.save_dir = save_dir
        self.use_dueling = use_dueling

        # Setup device (prioritize CUDA, then MPS, then CPU)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else self.device)

        # Use Dueling DQN architecture (better performance)
        NetworkClass = DuelingDQN if use_dueling else DQN
        self.policy_net = NetworkClass(input_dim, output_dim).to(self.device)
        self.target_net = NetworkClass(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is only for prediction, not training

        # Hyperparameters
        self.batch_size = 64  # Increased for better stability
        self.gamma = 0.99  # Discount factor

        # Optimizer (Increased learning rate)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=2.5e-4, amsgrad=True)

        # Memory with prioritized replay
        storage = LazyMemmapStorage(max_size=100000)
        self.memory = PrioritizedReplayBuffer(
            alpha=0.6,  # Prioritization exponent
            beta=0.4,  # Importance sampling exponent (annealed to 1.0)
            storage=storage,
            batch_size=self.batch_size
        )

        # Beta annealing parameters
        self.beta_start = 0.4
        self.beta_frames = 100000

        # Exploration settings (Improved decay)
        self.exploration_rate = 1.0
        self.exploration_decay = 0.99999  # Faster decay
        self.exploration_min = 0.01  # Lower minimum
        self.curr_step = 0

        # Training parameters
        self.burnin = 5e3  # Start learning sooner (reduced from 1e4)
        self.learn_every = 4  # Train every 4 steps
        self.sync_every = 1e4  # Sync target network every 10k steps

        # Tracking
        self.grad_norms = []  # Track gradient norms for diagnostics

    def act(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current state observation

        Returns:
            int: Action index to take
        """
        # EXPLORE: Random action
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT: Best action from policy network
        else:
            state = torch.tensor(np.array(state), device=self.device).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                action_idx = self.policy_net(state).argmax(dim=1).item()

        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)
        self.curr_step += 1

        return action_idx

    def cache(self, states, actions, next_states, rewards, done):
        """
        Store experience in prioritized replay buffer.

        Args:
            states: Current state
            actions: Action taken
            next_states: Next state
            rewards: Reward received
            done: Whether episode terminated
        """
        # Convert to tensors
        state = torch.from_numpy(states).to(self.device)
        next_state = torch.from_numpy(next_states).to(self.device)
        action = torch.tensor([actions], device=self.device)
        reward = torch.tensor([rewards], device=self.device)
        done = torch.tensor([done], device=self.device)

        # Create TensorDict for TorchRL compatibility
        data = TensorDict({
            'state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward,
            'done': done
        }, batch_size=[])

        self.memory.add(data)

    def recall(self):
        """
        Sample a batch of experiences from prioritized replay buffer.

        Returns:
            tuple: (states, actions, next_states, rewards, dones, indices, weights)
                   or None if buffer doesn't have enough samples
        """
        # Request samples with priority info
        samples, info = self.memory.sample(self.batch_size, return_info=True)

        indices = info['index']
        weights = info.get('_weight', torch.ones(self.batch_size))

        # Extract from TensorDict
        states = samples['state']
        actions = samples['action']
        next_states = samples['next_state']
        rewards = samples['reward']
        done = samples['done']

        return (
            states.to(self.device),
            actions.to(self.device),
            next_states.to(self.device),
            rewards.to(self.device),
            done.to(self.device),
            indices,
            weights.to(self.device)
        )

    def learn(self):
        """
        Update the policy network using prioritized experience replay.

        Implements:
        - Double DQN for stable Q-value estimation
        - Importance sampling weights
        - Beta annealing for bias correction
        - Gradient clipping

        Returns:
            tuple: (mean Q-value, loss) or (None, None) if not learning this step
        """
        # 1. Sync Target Net (Periodically)
        if self.curr_step % self.sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # 2. Check if we have enough memory to start learning
        if self.curr_step < self.burnin:
            return None, None

        # 3. Learn only every few steps (Stability)
        if self.curr_step % self.learn_every != 0:
            return None, None

        # 4. Anneal beta from 0.4 to 1.0 (bias correction)
        beta_progress = min(1.0, self.curr_step / self.beta_frames)
        current_beta = self.beta_start + beta_progress * (1.0 - self.beta_start)
        self.memory._beta = current_beta  # Update buffer's beta

        # 5. Sample from Memory
        sample = self.recall()
        if sample is None:
            return None, None

        state, action, next_state, reward, done, indices, weights = sample

        # Normalize to [0,1]
        state = state.float() / 255.0
        next_state = next_state.float() / 255.0

        # 6. Calculate TD estimates (what we predicted)
        td_est = self.policy_net(state).gather(1, action)

        # 7. Calculate TD targets (what we should have predicted)
        with torch.no_grad():
            # Double DQN: Policy net selects action, target net evaluates it
            best_action = self.policy_net(next_state).argmax(1).unsqueeze(1)
            next_state_values = self.target_net(next_state).gather(1, best_action)
            td_tgt = (reward + (1 - done.float()) * self.gamma * next_state_values)

        # 8. Calculate weighted loss with importance sampling
        elementwise_loss = F.smooth_l1_loss(td_est, td_tgt, reduction='none')
        loss = (elementwise_loss * weights.unsqueeze(1)).mean()

        # 9. Backpropagate
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients and track norm for diagnostics
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0  # Increased from 1.0 for more flexibility
        )
        self.grad_norms.append(grad_norm.item())

        # Warning if gradients are exploding
        if grad_norm > 5.0 and self.curr_step % 1000 == 0:
            print(f"Warning: High gradient norm: {grad_norm:.2f}")

        self.optimizer.step()

        # 10. Update priorities in buffer based on TD-errors
        with torch.no_grad():
            td_errors = torch.abs(td_tgt - td_est).detach().cpu().flatten()
            new_priorities = (td_errors + 1e-6).clamp(max=1e2)  # Prevent 0 and overflow

        self.memory.update_priority(indices, new_priorities)

        return td_est.mean().item(), loss.item()

    def save_model(self, filepath, episode=None, metadata=None):
        """
        Save the model checkpoint.

        Args:
            filepath: Path to save the model
            episode: Current episode number (optional)
            metadata: Additional metadata to save (optional)
        """
        save_dict = {
            'model_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'curr_step': self.curr_step,
            'architecture': 'dueling_dqn' if self.use_dueling else 'dqn',
        }

        if episode is not None:
            save_dict['episode'] = episode

        if metadata is not None:
            save_dict['metadata'] = metadata

        torch.save(save_dict, filepath)
        print(f"Model saved: {filepath}")

    def load_model(self, filepath, load_optimizer=True):
        """
        Load a model checkpoint.

        Args:
            filepath: Path to the saved model
            load_optimizer: Whether to load optimizer state (default: True)

        Returns:
            dict: Checkpoint dictionary with metadata
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'exploration_rate' in checkpoint:
            self.exploration_rate = checkpoint['exploration_rate']

        if 'curr_step' in checkpoint:
            self.curr_step = checkpoint['curr_step']

        print(f"Model loaded: {filepath}")
        return checkpoint
