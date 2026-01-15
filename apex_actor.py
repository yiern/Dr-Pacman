"""
APEX Actor - Asynchronous Experience Collection for APEX-DQN

This module implements the actor process for APEX (Asynchronous Prioritized
Experience Replay) architecture. Actors collect experiences by playing the game
in parallel and send them to a centralized learner.
"""

import torch
import numpy as np
import queue

import gymnasium as gym
import ale_py
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation

from dqn_networks import DuelingDQN
from pacman_reward_wrapper import PacmanRewardWrapper


class APEXActor:
    """
    Actor process for APEX architecture.

    Actors collect experiences by playing the game with an epsilon-greedy policy.
    Each actor has slightly different exploration parameters for diversity.
    Periodically syncs weights from the centralized learner.
    """

    def __init__(self, actor_id, experience_queue, weight_queue, frameskip=4):
        """
        Initialize the APEX actor.

        Args:
            actor_id: Unique identifier for this actor
            experience_queue: Shared queue to send experiences to learner
            weight_queue: Queue to receive updated weights from learner
            frameskip: Number of frames to skip in environment
        """
        self.actor_id = actor_id
        self.experience_queue = experience_queue
        self.weight_queue = weight_queue
        self.frameskip = frameskip

    def run(self, episodes_per_actor, exploration_offset=0.0):
        """
        Run the actor to collect experiences.

        Args:
            episodes_per_actor: Number of episodes to collect
            exploration_offset: Offset for exploration rate (for diversity)
        """
        # Setup environment
        gym.register_envs(ale_py)
        env = gym.make('ALE/Pacman-v5', render_mode='rgb_array', frameskip=self.frameskip)
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env)
        env = FrameStackObservation(env, 4)
        env = PacmanRewardWrapper(env)

        # Set up a policy network (CPU only for multiprocessing compatibility)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        policy_net = DuelingDQN(input_shape= env.observation_space.shape, num_actions= env.action_space.n).to(device)
        policy_net.eval()

        # Actor-specific exploration (for diversity)
        exploration_rate = 1.0 - exploration_offset
        exploration_decay = 0.99999
        exploration_min = 0.01 + (self.actor_id * 0.02)  # Each actor has different minimum

        experiences_collected = 0

        print(f"[Actor {self.actor_id}] Started (ε: {exploration_rate:.3f} -> {exploration_min:.3f})")

        for episode in range(episodes_per_actor):
            state, _ = env.reset()
            episode_reward = 0

            # Sync weights from learner
            try:
                if not self.weight_queue.empty():
                    new_weights = self.weight_queue.get_nowait()
                    policy_net.load_state_dict(new_weights)
            except queue.Empty:
                pass

            # Collect episode
            while True:
                # Select action (epsilon-greedy)
                # Guess
                if np.random.rand() < exploration_rate:
                    action = env.action_space.sample()
                # Learned step
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(np.array(state), device=device).unsqueeze(0).float() / 255.0
                        action = policy_net(state_tensor).argmax(dim=1).item()

                # Step environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Send experience to learner
                try:
                    self.experience_queue.put_nowait({
                        'state': state,
                        'action': action,
                        'next_state': next_state,
                        'reward': reward,
                        'done': done
                    })
                    experiences_collected += 1
                except queue.Full:
                    pass  # Skip if queue full

                state = next_state
                episode_reward += reward
                exploration_rate = max(exploration_min, exploration_rate * exploration_decay)

                if done:
                    break

            # Progress update
            if (episode + 1) % 50 == 0:
                print(f"[Actor {self.actor_id}] Ep {episode + 1}/{episodes_per_actor}, "
                      f"Exp: {experiences_collected}, Reward: {episode_reward:.1f}")

        env.close()
        print(f"[Actor {self.actor_id}] Finished - {experiences_collected} experiences collected")


# Helper function to run an actor (for multiprocessing)
def run_actor(actor_id, experience_queue, weight_queue, episodes_per_actor,
              exploration_offset=0.0, frameskip=4):
    """
    Helper function to run an actor process.

    Args:
        actor_id: Unique identifier for this actor
        experience_queue: Shared queue to send experiences
        weight_queue: Queue to receive updated weights
        episodes_per_actor: Number of episodes to collect
        exploration_offset: Offset for exploration rate
        frameskip: Number of frames to skip
    """
    actor = APEXActor(actor_id, experience_queue, weight_queue, frameskip)
    actor.run(episodes_per_actor, exploration_offset)
