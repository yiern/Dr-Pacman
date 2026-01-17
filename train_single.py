"""
Single Process DQN Training for Pac-Man

Simple synchronous training where the agent:
1. Collects an experience
2. Stores it in replay buffer
3. Learns from the buffer
4. Repeats

This is the traditional DQN approach without multiprocessing.
"""

import gymnasium as gym
import ale_py
import numpy as np
import torch
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation

from pacman_reward_wrapper import PacmanRewardWrapper
from rl_agent import RlAgent


def find_latest_checkpoint(save_dir):
    """
    Automatically find the best checkpoint to resume from.

    Priority:
    1. dqn_final.pth (highest - final trained model)
    2. dqn_episode_*.pth with highest episode number
    3. None (no checkpoints found)

    Args:
        save_dir: Directory to search for checkpoints

    Returns:
        str: Path to checkpoint, or None if no checkpoints found
    """
    import os
    import re

    if not os.path.exists(save_dir):
        return None

    # Check for dqn_final.pth first (highest priority)
    final_path = os.path.join(save_dir, 'dqn_final.pth')
    if os.path.isfile(final_path):
        return final_path

    # Find latest dqn_episode_*.pth
    episode_pattern = re.compile(r'dqn_episode_(\d+)\.pth')
    episode_checkpoints = []

    for filename in os.listdir(save_dir):
        match = episode_pattern.match(filename)
        if match:
            episode_num = int(match.group(1))
            filepath = os.path.join(save_dir, filename)
            episode_checkpoints.append((episode_num, filepath))

    if episode_checkpoints:
        # Return checkpoint with highest episode number
        episode_checkpoints.sort(reverse=True)
        latest_episode, latest_path = episode_checkpoints[0]
        return latest_path

    return None


def train_single_process(
    episodes=1000,
    frameskip=4,
    save_dir="saved_models",
    save_every=250,
    print_every=250,
    checkpoint=None
):
    """
    Train a Pac-Man agent using single-process DQN.

    Args:
        episodes: Number of ADDITIONAL episodes to train
        frameskip: Number of frames to skip in environment
        save_dir: Directory to save model checkpoints
        save_every: Save model every N episodes
        print_every: Print progress every N episodes
        checkpoint: Path to checkpoint file to resume training.
                   - None (default): Auto-discover best checkpoint in save_dir
                   - str: Explicit path to specific checkpoint
                   - False: Force fresh start (skip auto-discovery)
    """
    print("=" * 80)
    print("🎮 SINGLE PROCESS DQN TRAINING")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  Frameskip: {frameskip}")
    print(f"  Save directory: {save_dir}")
    if checkpoint:
        print(f"  Checkpoint: {checkpoint}")
        print(f"  Resume mode: ENABLED")
    print("=" * 80)
    print()

    # Setup environment
    gym.register_envs(ale_py)
    env = gym.make('ALE/Pacman-v5', render_mode='rgb_array', frameskip=frameskip)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, 4)
    env = PacmanRewardWrapper(env)

    # Get dimensions
    input_dim = env.observation_space.shape
    output_dim = env.action_space.n

    print(f"Environment setup:")
    print(f"  Input dimensions: {input_dim}")
    print(f"  Number of actions: {output_dim}")
    print()

    # Create agent
    agent = RlAgent(input_dim, output_dim, save_dir=save_dir, use_dueling=True)
    print(f"Agent initialized on device: {agent.device}")
    print(f"  Architecture: {'Dueling DQN' if agent.use_dueling else 'DQN'}")
    print(f"  Exploration rate: {agent.exploration_rate:.3f} -> {agent.exploration_min:.3f}")
    print(f"  Burnin: {int(agent.burnin)} steps")
    print()

    # Auto-discover checkpoint if not explicitly provided
    if checkpoint is None:
        auto_checkpoint = find_latest_checkpoint(save_dir)
        if auto_checkpoint:
            checkpoint = auto_checkpoint
            print("=" * 80)
            print("🔍 AUTO-DISCOVERY: Found existing checkpoint")
            print("=" * 80)
            print(f"  ✓ Auto-selected: {checkpoint}")
            print("  ℹ To start fresh, delete checkpoints or use checkpoint=False")
            print("=" * 80)
            print()

    # Load checkpoint if provided (manual or auto-discovered)
    # Skip if checkpoint=False (explicit fresh start)
    if checkpoint and checkpoint is not False:
        print("=" * 80)
        print("📦 Loading checkpoint for continued training...")
        print("=" * 80)

        import os
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        try:
            # Load using agent's method - returns checkpoint dictionary
            checkpoint_data = agent.load_model(checkpoint, load_optimizer=True)

            # Display what was loaded from checkpoint
            if 'episode' in checkpoint_data:
                print(f"  ✓ Loaded checkpoint from episode: {checkpoint_data['episode']}")

            # Display restored state
            print(f"  ✓ Restored exploration rate: {agent.exploration_rate:.4f}")
            print(f"  ✓ Restored curr_step: {agent.curr_step}")

            # Display previous training metadata
            if 'metadata' in checkpoint_data:
                metadata = checkpoint_data['metadata']
                if 'avg_reward_100' in metadata:
                    print(f"  ✓ Previous avg reward (100 ep): {metadata['avg_reward_100']:.2f}")
                if 'total_steps' in metadata:
                    print(f"  ✓ Previous total steps: {metadata['total_steps']}")

            print()
            print("  ℹ Starting NEW training session from episode 1")
            print("  ℹ Note: Replay buffer is empty and will refill during burnin")
            print(f"  ℹ Burnin period: {int(agent.burnin)} steps")

        except Exception as e:
            print(f"  ✗ Error loading checkpoint: {e}")
            raise

        print("=" * 80)
        print()

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    recent_rewards = []
    total_steps = 0

    print("🚀 Starting training...")
    print()

    # Training loop
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        episode_q_values = []

        while True:
            # Agent chooses action
            action = agent.act(state)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience in replay buffer
            agent.cache(
                states=np.array(state),
                actions=action,
                next_states=np.array(next_state),
                rewards=reward,
                done=done
            )

            # Learn from experience
            q_value, loss = agent.learn()

            if q_value is not None:
                episode_q_values.append(q_value)
            if loss is not None:
                episode_losses.append(loss)

            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            if done:
                break

        # Track episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        # Print progress
        if episode % print_every == 0:
            avg_reward = np.mean(recent_rewards)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            avg_q = np.mean(episode_q_values) if episode_q_values else 0

            print(f"Episode {episode:4d}/{episodes} | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Avg(100): {avg_reward:7.1f} | "
                  f"Steps: {episode_length:4d} | "
                  f"ε: {agent.exploration_rate:.3f} | "
                  f"Q: {avg_q:6.2f} | "
                  f"Loss: {avg_loss:.4f}")

        # Save model
        if episode % save_every == 0 and episode >= 500:
            import os
            os.makedirs(save_dir, exist_ok=True)
            filepath = f"{save_dir}/dqn_episode_{episode}.pth"
            agent.save_model(
                filepath,
                episode=episode,
                metadata={
                    'total_steps': total_steps,
                    'avg_reward_100': np.mean(recent_rewards),
                    'episode_reward': episode_reward
                }
            )

    # Save final model
    import os
    os.makedirs(save_dir, exist_ok=True)
    final_path = f"{save_dir}/dqn_final.pth"
    agent.save_model(
        final_path,
        episode=episodes,
        metadata={
            'total_steps': total_steps,
            'avg_reward_100': np.mean(recent_rewards),
            'all_rewards': episode_rewards
        }
    )

    env.close()

    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total steps: {total_steps}")
    print(f"Average reward (last 100): {np.mean(recent_rewards):.1f}")
    print(f"Best episode reward: {max(episode_rewards):.1f}")
    print(f"Final model saved: {final_path}")
    print()

    return episode_rewards, episode_lengths


def main():
    """Main entry point for training."""

    # Quick test configuration (uncomment to test)
    # train_single_process(
    #     episodes=10,
    #     frameskip=4,
    #     save_dir="saved_models",
    #     save_every=5,
    #     print_every=2
    # )

    # Example 1: Auto-discovery (NEW - finds dqn_final.pth automatically)
    # train_single_process(
    #     episodes=1000
    # )
    # Automatically loads dqn_final.pth if it exists

    # Example 2: Force fresh start (ignore existing checkpoints)
    # train_single_process(
    #     episodes=1000,
    #     checkpoint=False  # Skip auto-discovery
    # )

    # Example 3: Explicit checkpoint (manual selection)
    # train_single_process(
    #     episodes=1000,
    #     checkpoint="saved_models/dqn_episode_500.pth"
    # )

    # Full training configuration

    train_single_process(
        episodes=1000,
        frameskip=4,
        save_dir="saved_models",
        save_every=100,
        print_every=10
    )


if __name__ == '__main__':
    main()
