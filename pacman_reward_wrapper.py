"""
Pacman Reward Wrapper - Balanced reward shaping for Pac-Man environment

This wrapper modifies the reward function to provide better training signals:
- Makes dots more valuable
- Reduces death penalty to be recoverable
- Adds small time penalty to encourage action
"""

import gymnasium as gym


class PacmanRewardWrapper(gym.Wrapper):
    """
    REBALANCED reward shaping for Pacman.

    Key insight: Death must be recoverable through good play.
    """

    def __init__(self, env):
        """
        Initialize the reward wrapper.

        Args:
            env: Gymnasium environment to wrap
        """
        super().__init__(env)
        self.lives = 0
        self.steps = 0

    def reset(self, **kwargs):
        """
        Reset the environment and tracking variables.

        Args:
            **kwargs: Additional arguments to pass to environment reset

        Returns:
            tuple: (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get('lives', 3)
        self.steps = 0
        return obs, info

    def step(self, action):
        """
        Take a step in the environment with shaped rewards.

        Args:
            action: Action to take

        Returns:
            tuple: (observation, shaped_reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = info.get('lives', 0)
        self.steps += 1

        # --- REBALANCED REWARD SHAPING ---
        shaped_reward = 0

        # 1. BASE REWARDS: Make dots MUCH more valuable
        # Base game: dot=10, power pellet=50, ghost=200-1600
        if reward > 0:
            shaped_reward = reward * 0.5  # Increased from 0.1
            # Now: dot=5, pellet=25, ghost=100-800

        # 2. SMALL TIME PENALTY (not bonus!)
        # Penalize doing nothing, but make it small
        elif reward == 0:
            shaped_reward = -0.001  # Tiny penalty for not progressing

        # 3. DEATH PENALTY: Reduced to be recoverable
        if current_lives < self.lives:
            shaped_reward -= 5.0  # Reduced from -10.0
            self.lives = current_lives

        # NEW BALANCE:
        # - Eating 1 dot: +5.0 (recovers from death!)
        # - Death: -5.0
        # - 400 steps with no dots: -0.4
        # - Agent can break even by eating dots after death

        return obs, shaped_reward, terminated, truncated, info
