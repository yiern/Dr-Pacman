"""
APEX Training Script - Asynchronous Prioritized Experience Replay

This script trains a Pac-Man agent using the APEX architecture with:
- Multiple parallel actors collecting experiences
- Centralized learner training the network
- Prioritized experience replay

Usage:
    python train_apex.py
"""

import multiprocessing as mp
from multiprocessing import Queue, Process
import time
import torch

import gymnasium as gym
import ale_py
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation

from pacman_reward_wrapper import PacmanRewardWrapper
from apex_actor import APEXActor
from apex_learner import APEXLearner


def train_apex(
    num_actors=4,
    episodes_per_actor=500,
    frameskip=4,
    save_dir="saved_models",
    checkpoint_every=10000,
    max_queue_size=10000):
    """
    Train a Pac-Man agent using APEX architecture.

    Args:
        num_actors: Number of parallel actors collecting experiences
        episodes_per_actor: Episodes each actor should play
        frameskip: Number of frames to skip in environment
        save_dir: Directory to save model checkpoints
        checkpoint_every: Save checkpoint every N experiences
        max_queue_size: Maximum size of experience queue
    """
    print("🚀 APEX TRAINING - Asynchronous Prioritized Experience Replay")

    print(f"Configuration:\n  Actors: {num_actors}\n  Episodes per actor: {episodes_per_actor}\n  Frameskip: {frameskip}\n  Save directory: {save_dir}\n" + "=" * 80 + "\n")

    # Create shared queues
    experience_queue = Queue(maxsize=max_queue_size)
    weight_queues = [Queue(maxsize=2) for _ in range(num_actors)]

    # Get environment dimensions
    gym.register_envs(ale_py)
    temp_env = gym.make('ALE/Pacman-v5', frameskip=frameskip)
    temp_env = ResizeObservation(temp_env, (84, 84))
    temp_env = GrayscaleObservation(temp_env)
    temp_env = FrameStackObservation(temp_env, 4)
    input_dim = temp_env.observation_space.shape
    output_dim = temp_env.action_space.n
    temp_env.close()

    print(f"Environment setup:")
    print(f"  Input dimensions: {input_dim}")
    print(f"  Number of actions: {output_dim}")
    print()

    # Estimate total experiences
    total_experiences = num_actors * episodes_per_actor * 100  # Rough estimate

    # Create learner (FIXED: doesn't create agent until run() is called)
    learner = APEXLearner(
        input_dim,
        output_dim,
        experience_queue,
        weight_queues,
        save_dir=save_dir,
        checkpoint_every=checkpoint_every
    )

    learner_process = Process(target=learner.run, args=(total_experiences,))

    # Create actors
    actor_processes = []
    for i in range(num_actors):
        actor = APEXActor(i, experience_queue, weight_queues[i], frameskip)
        exploration_offset = i * 0.1  # Each actor has different exploration
        p = Process(target=actor.run, args=(episodes_per_actor, exploration_offset))
        actor_processes.append(p)

    # Start processes
    print("🚀 Starting processes...")
    print()

    # Start learner first
    learner_process.start()
    time.sleep(2)  # Give learner time to initialize

    # Start actors
    for p in actor_processes:
        p.start()
        time.sleep(0.5)

    # Wait for actors to complete
    print("⏳ Actors collecting experiences...")
    for i, p in enumerate(actor_processes):
        p.join()
        print(f"[Main] Actor {i} completed")

    # Give learner time to process remaining experiences
    print("[Main] Waiting for learner to finish processing...")
    time.sleep(10)

    # Terminate learner
    learner_process.terminate()
    learner_process.join()

    print()
    print("=" * 80)
    print("✅ APEX TRAINING COMPLETE!")
    print("=" * 80)
    print(f"📁 Models saved in: {save_dir}/")
    print(f"   - apex_final.pth (final model)")
    print(f"   - apex_checkpoint_*.pth (checkpoints)")
    print()


def main():
    """Main entry point for APEX training."""
    # Test with small numbers first
    print("Starting APEX training with test parameters...")
    print("(Use larger values for full training)")
    print()

    # train_apex(
    #     num_actors=2,           # Start with 2 actors for testing
    #     episodes_per_actor=10,  # 10 episodes for quick test
    #     frameskip=4,
    #     save_dir="saved_models",
    #     checkpoint_every=1000
    # )

    #For full training, use:
    train_apex(
        num_actors=4,
        episodes_per_actor=500,
        frameskip=4,
        save_dir="saved_models",
        checkpoint_every=10000
    )


if __name__ == '__main__':
    # REQUIRED: multiprocessing guard for Windows/Mac
    mp.set_start_method('spawn', force=True)
    main()
