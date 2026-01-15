"""
Evaluate Trained Agent - Test and watch your trained Pac-Man agent

This script loads a trained model and evaluates its performance by:
- Running test episodes
- Computing average reward, episode length, and success metrics
- Optionally rendering the game to watch the agent play
"""

import torch
import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation

from rl_agent import RlAgent
from pacman_reward_wrapper import PacmanRewardWrapper


def evaluate_agent(
    model_path="saved_models/apex_final.pth",
    num_episodes=10,
    render=False,
    exploration_rate=0.01,  # Low epsilon for evaluation
    frameskip=4
):
    """
    Evaluate a trained agent.

    Args:
        model_path: Path to the saved model checkpoint
        num_episodes: Number of episodes to evaluate
        render: Whether to render the game (slower but you can watch)
        exploration_rate: Epsilon for evaluation (usually small, like 0.01)
        frameskip: Number of frames to skip

    Returns:
        dict: Evaluation metrics (avg_reward, avg_length, best_reward, etc.)
    """
    print("=" * 80)
    print("🎮 EVALUATING TRAINED AGENT")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    print(f"Exploration rate: {exploration_rate}")
    print("=" * 80)
    print()

    # Setup environment
    gym.register_envs(ale_py)
    render_mode = 'human' if render else 'rgb_array'
    env = gym.make('ALE/Pacman-v5', render_mode=render_mode, frameskip=frameskip)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, 4)
    env = PacmanRewardWrapper(env)

    # Create agent
    agent = RlAgent(
        input_dim=env.observation_space.shape,
        output_dim=env.action_space.n,
        use_dueling=True
    )

    # Load trained model
    try:
        checkpoint = agent.load_model(model_path, load_optimizer=False)
        print(f"✅ Model loaded successfully")
        if 'experiences_processed' in checkpoint:
            print(f"   Experiences processed: {checkpoint['experiences_processed']}")
        if 'training_steps' in checkpoint:
            print(f"   Training steps: {checkpoint['training_steps']}")
        print()
    except FileNotFoundError:
        print(f"❌ Error: Model not found at {model_path}")
        print("   Make sure you've run train_apex.py first!")
        return None

    # Set evaluation mode
    agent.policy_net.eval()
    agent.exploration_rate = exploration_rate  # Low epsilon for evaluation

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    episode_scores = []  # Raw game scores

    # Run evaluation episodes
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        raw_score = 0

        while True:
            # Select action (mostly greedy, small epsilon)
            action = agent.act(state)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            raw_score = info.get('score', 0)  # Get raw Atari score

            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(raw_score)

        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward: {episode_reward:.1f}, "
              f"Length: {episode_length}, "
              f"Score: {raw_score}")

    env.close()

    # Compute statistics
    metrics = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'avg_score': np.mean(episode_scores),
        'max_score': np.max(episode_scores),
    }

    # Print summary
    print()
    print("=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Reward Range: [{metrics['min_reward']:.1f}, {metrics['max_reward']:.1f}]")
    print(f"Average Episode Length: {metrics['avg_length']:.1f} steps")
    print(f"Average Game Score: {metrics['avg_score']:.1f}")
    print(f"Best Game Score: {metrics['max_score']:.0f}")
    print("=" * 80)

    return metrics


def compare_models(model_paths, num_episodes=10):
    """
    Compare multiple trained models.

    Args:
        model_paths: List of paths to model checkpoints
        num_episodes: Number of episodes to evaluate each model

    Returns:
        dict: Comparison results
    """
    print("=" * 80)
    print("🔍 COMPARING MODELS")
    print("=" * 80)
    print()

    results = {}

    for model_path in model_paths:
        print(f"\nEvaluating: {model_path}")
        print("-" * 80)
        metrics = evaluate_agent(
            model_path=model_path,
            num_episodes=num_episodes,
            render=False
        )
        if metrics:
            results[model_path] = metrics

    # Print comparison table
    print()
    print("=" * 80)
    print("📊 COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<40} {'Avg Reward':<15} {'Avg Score':<15}")
    print("-" * 80)

    for model_path, metrics in results.items():
        model_name = model_path.split('/')[-1]  # Get filename only
        print(f"{model_name:<40} "
              f"{metrics['avg_reward']:>7.2f} ± {metrics['std_reward']:<5.2f} "
              f"{metrics['avg_score']:>10.1f}")

    print("=" * 80)

    return results


def watch_agent_play(
    model_path="saved_models/apex_final.pth",
    num_episodes=5,
    exploration_rate=0.0,  # Fully greedy
    frameskip=4,
    slow_mode=False
):
    """
    Watch the trained agent play (with rendering).

    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to watch
        exploration_rate: Epsilon (0.0 for fully greedy)
        frameskip: Frame skip
        slow_mode: If True, adds delay to slow down gameplay
    """
    import time

    print("=" * 80)
    print("👀 WATCHING AGENT PLAY")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Close the game window to stop early")
    print("=" * 80)
    print()

    metrics = evaluate_agent(
        model_path=model_path,
        num_episodes=num_episodes,
        render=True,
        exploration_rate=exploration_rate,
        frameskip=frameskip
    )

    if slow_mode and metrics:
        time.sleep(0.1)  # Slow down for better viewing

    return metrics


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained Pac-Man agent')
    parser.add_argument('--model', type=str, default='saved_models/apex_final.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                        help='Render the game (watch agent play)')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare multiple models')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='Exploration rate (default: 0.01)')

    args = parser.parse_args()

    if args.compare:
        # Compare multiple models
        compare_models(args.compare, num_episodes=args.episodes)
    elif args.render:
        # Watch agent play
        watch_agent_play(
            model_path=args.model,
            num_episodes=args.episodes,
            exploration_rate=args.epsilon
        )
    else:
        # Standard evaluation
        evaluate_agent(
            model_path=args.model,
            num_episodes=args.episodes,
            render=False,
            exploration_rate=args.epsilon
        )


if __name__ == '__main__':
    # Example usage:

    # 1. Quick evaluation (no rendering)
    print("Running quick evaluation...")
    evaluate_agent(
        model_path="saved_models/apex_final.pth",
        num_episodes=10,
        render=False
    )

    # 2. Watch agent play (uncomment to enable)
    # print("\nNow watching agent play...")
    # watch_agent_play(
    #     model_path="saved_models/apex_final.pth",
    #     num_episodes=3,
    #     exploration_rate=0.0  # Fully greedy
    # )

    # 3. Compare checkpoints (uncomment to enable)
    # compare_models([
    #     "saved_models/apex_checkpoint_10000.pth",
    #     "saved_models/apex_checkpoint_50000.pth",
    #     "saved_models/apex_final.pth"
    # ], num_episodes=10)
