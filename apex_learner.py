"""
APEX Learner - Centralized Training for APEX-DQN

This module implements the learner process for APEX (Asynchronous Prioritized
Experience Replay) architecture. The learner receives experiences from actors,
trains the network, and broadcasts updated weights back to actors.

IMPORTANT: The RlAgent is created INSIDE the run() method, not in __init__().
This is crucial to avoid LazyMemmapStorage pickling errors with multiprocessing.
"""

import torch
import numpy as np
import queue
import time
import os

from rl_agent import RlAgent


class APEXLearner:
    """
    Learner process for APEX architecture.

    Receives experiences from multiple actors via a shared queue, trains the
    network using prioritized experience replay, and broadcasts updated weights
    back to actors.

    FIXED: Creates RlAgent INSIDE run() method to avoid multiprocessing errors.
    """

    def __init__(self, input_dim, output_dim, experience_queue, weight_queues,
                 save_dir="saved_models", checkpoint_every=10000):
        """
        Initialize the APEX learner (stores parameters only).

        IMPORTANT: Does NOT create the RlAgent here to avoid pickling errors.
        The agent is created inside run() method when the subprocess starts.

        Args:
            input_dim: Input dimensions for the network
            output_dim: Number of actions
            experience_queue: Shared queue to receive experiences from actors
            weight_queues: List of queues to broadcast weights to actors
            save_dir: Directory to save checkpoints
            checkpoint_every: Save checkpoint every N experiences
        """
        # Store parameters only (KEY FIX - don't create agent here!)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.experience_queue = experience_queue
        self.weight_queues = weight_queues
        self.save_dir = save_dir
        self.checkpoint_every = checkpoint_every

    def run(self, total_experiences_target):
        """
        Train the network on experiences from the queue.

        Creates the RlAgent HERE (inside subprocess) to avoid LazyMemmapStorage
        pickling errors with multiprocessing.

        Args:
            total_experiences_target: Total number of experiences to process
        """
        # Create agent HERE in subprocess (KEY FIX!)
        self.agent = RlAgent(self.input_dim, self.output_dim, use_dueling=True)

        # Force CPU if MPS (multiprocessing incompatible)
        if self.agent.device.type == 'mps':
            print("⚠️  [Learner] MPS detected - forcing CPU for multiprocessing")
            self.agent.device = torch.device('cpu')
            self.agent.policy_net = self.agent.policy_net.to('cpu')
            self.agent.target_net = self.agent.target_net.to('cpu')

        print(f"[Learner] Started on {self.agent.device}")
        os.makedirs(self.save_dir, exist_ok=True)

        # Training metrics
        experiences_processed = 0
        training_steps = 0
        last_weight_broadcast = 0
        last_checkpoint = 0
        q_values = []
        losses = []

        # Training loop
        while experiences_processed < total_experiences_target:
            try:
                # Get experience from queue
                experience = self.experience_queue.get(timeout=1)

                # Add to replay buffer
                self.agent.cache(
                    experience['state'],
                    experience['action'],
                    experience['next_state'],
                    experience['reward'],
                    experience['done']
                )
                experiences_processed += 1

                # Train if we have enough experiences
                if experiences_processed >= self.agent.burnin:
                    q, loss = self.agent.learn()
                    if q is not None:
                        q_values.append(q)
                    if loss is not None:
                        losses.append(loss)
                    training_steps += 1

                # Broadcast weights to actors
                if training_steps - last_weight_broadcast >= 100:
                    weights = self.agent.policy_net.state_dict()
                    for wq in self.weight_queues:
                        try:
                            # Clear old weights
                            while not wq.empty():
                                wq.get_nowait()
                            # Send new weights
                            wq.put_nowait(weights)
                        except queue.Full:
                            pass
                    last_weight_broadcast = training_steps

                # Progress update
                if experiences_processed % 1000 == 0:
                    avg_q = np.mean(q_values[-100:]) if q_values else 0
                    avg_loss = np.mean(losses[-100:]) if losses else 0
                    print(f"[Learner] Exp: {experiences_processed}/{total_experiences_target}, "
                          f"Q: {avg_q:.2f}, Loss: {avg_loss:.4f}")

                # Checkpoint
                if experiences_processed - last_checkpoint >= self.checkpoint_every:
                    self._save_checkpoint(experiences_processed, training_steps)
                    last_checkpoint = experiences_processed

            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"[Learner] Error: {e}")
                import traceback
                traceback.print_exc()
                break

        # Save final model
        print(f"[Learner] Training complete - {training_steps} training steps")
        self._save_final_model(experiences_processed, training_steps)

    def _save_checkpoint(self, experiences, steps):
        """
        Save a checkpoint of the model.

        Args:
            experiences: Number of experiences processed
            steps: Number of training steps taken
        """
        path = os.path.join(self.save_dir, f"apex_checkpoint_{experiences}.pth")
        torch.save({
            'experiences_processed': experiences,
            'training_steps': steps,
            'model_state_dict': self.agent.policy_net.state_dict(),
            'target_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'exploration_rate': self.agent.exploration_rate,
            'architecture': 'dueling_dqn_apex',
        }, path)
        print(f"[Learner] 💾 Checkpoint saved: {path}")

    def _save_final_model(self, experiences, steps):
        """
        Save the final trained model.

        Args:
            experiences: Number of experiences processed
            steps: Number of training steps taken
        """
        path = os.path.join(self.save_dir, "apex_final.pth")
        torch.save({
            'experiences_processed': experiences,
            'training_steps': steps,
            'model_state_dict': self.agent.policy_net.state_dict(),
            'target_state_dict': self.agent.target_net.state_dict(),
            'architecture': 'dueling_dqn_apex',
        }, path)
        print(f"[Learner] 🎉 Final model saved: {path}")


# Helper function to run learner (for multiprocessing)
def run_learner(input_dim, output_dim, experience_queue, weight_queues,
                total_experiences_target, save_dir="saved_models", checkpoint_every=10000):
    """
    Helper function to run the learner process.

    Args:
        input_dim: Input dimensions for the network
        output_dim: Number of actions
        experience_queue: Shared queue to receive experiences
        weight_queues: List of queues to broadcast weights
        total_experiences_target: Total experiences to process
        save_dir: Directory to save checkpoints
        checkpoint_every: Save checkpoint every N experiences
    """
    learner = APEXLearner(
        input_dim, output_dim, experience_queue, weight_queues,
        save_dir, checkpoint_every
    )
    learner.run(total_experiences_target)
