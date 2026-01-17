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


def train_single_process(
    episodes=1000,
    frameskip=4,
    save_dir="saved_models",
    save_every=100,
    print_every=250
):
    """
    Train a Pac-Man agent using single-process DQN.

    Args:
        episodes: Number of episodes to train
        frameskip: Number of frames to skip in environment
        save_dir: Directory to save model checkpoints
        save_every: Save model every N episodes
        print_every: Print progress every N episodes
    """
    print("=" * 80)
    print("🎮 SINGLE PROCESS DQN TRAINING")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  Frameskip: {frameskip}")
    print(f"  Save directory: {save_dir}")
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
        if episode % save_every == 0:
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
