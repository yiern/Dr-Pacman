import torch
from dqn_networks import DuelingDQN
import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    ResizeObservation,
    GrayscaleObservation,
    FrameStackObservation,
)

from pacman_reward_wrapper import PacmanRewardWrapper


class PlayAgent:
    def __init__(self):
        env = gym.make("ALE/Pacman-v5", render_mode="human")
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env)
        env = FrameStackObservation(env, 4)
        env = PacmanRewardWrapper(env)
        self.env = env

        num_actions = env.action_space.n
        model = DuelingDQN(
            input_shape=env.observation_space.shape, num_actions=num_actions
        )
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.policy_net = model.to(self.device)

    def play(self, checkpoint_path=None, num_episodes=5):
        """
        Play the game using trained weights.

        Args:
            checkpoint_path: Path to .pth checkpoint file
            num_episodes: Number of episodes to play
        """
        if checkpoint_path is None:
            print("Error: No checkpoint path provided")
            return None

        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "policy_net_state_dict" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        elif "model_state_dict" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.policy_net.load_state_dict(checkpoint)
        self.policy_net.eval()

        episode_scores = []
        episode_lengths = []
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor)
                action = q_values.argmax(dim=-1).item()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
                steps += 1
                self.env.render()
                state = next_state
            episode_scores.append(total_reward)
            episode_lengths.append(steps)
            print(f"Episode {ep + 1}: reward={total_reward:.1f}, steps={steps}")

        if episode_scores:
            avg_score = sum(episode_scores) / len(episode_scores)
            max_score = max(episode_scores)
            avg_length = sum(episode_lengths) / len(episode_lengths)
            print(
                f"Summary: avg={avg_score:.1f}, max={max_score:.1f}, avg_len={avg_length:.1f}"
            )

        self.env.close()


if __name__ == "__main__":
    player = PlayAgent()
    player.play(checkpoint_path="saved_models/dqn_final.pth", num_episodes=3)
