"""
Deep Q-Network (DQN) Architectures

Contains two neural network architectures for Q-learning:
1. DQN - Standard Deep Q-Network
2. DuelingDQN - Dueling architecture that separates value and advantage streams
"""

import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Standard Deep Q-Network Architecture.

    This is the baseline DQN architecture with three convolutional layers
    followed by two fully connected layers.

    Basic CNN architecture to read the game's image

    """

    def __init__(self, input_shape, num_actions):
        """
        Initialize the DQN network.

        Args:
            input_shape: Shape of input (channels, height, width) or int for channels
            num_actions: Number of possible actions
        """
        super(DQN, self).__init__()

        channels = input_shape[0] if isinstance(input_shape, tuple) else 4

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network Architecture.

    Separates Q(s,a) into:
    - V(s): Value of being in state s
    - A(s,a): Advantage of taking action a in state s

    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

    """
    def __init__(self, input_shape, num_actions):
        """
        Initialize the Dueling DQN network.

        Args:
            input_shape: Shape of input (channels, height, width) or int for channels
            num_actions: Number of possible actions
        """
        super(DuelingDQN, self).__init__()

        channels = input_shape[0] if isinstance(input_shape, tuple) else 4

        # Shared convolutional feature extractor
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4)
        # Output: (32, 20, 20)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # Output: (64, 9, 9)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # Output: (64, 7, 7)

        # Flattened size: 64 * 7 * 7 = 3136

        # Value stream: V(s)
        # Estimates "how good is this state"
        self.value_fc = nn.Linear(3136, 512)
        self.value = nn.Linear(512, 1)  # Single scalar value

        # Advantage stream: A(s,a)
        # Estimates "how much better is action a compared to average"
        self.advantage_fc = nn.Linear(3136, 512)
        self.advantage = nn.Linear(512, num_actions)  # One value per action

    def forward(self, x):
        """
        Forward pass through the dueling network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Q-values for each action of shape (batch, num_actions)
        """
        # Shared feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten: (Batch, 64, 7, 7) -> (Batch, 3136)
        x = x.view(x.size(0), -1)

        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value(value)  # (Batch, 1)

        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)  # (Batch, num_actions)

        # Combine streams using the dueling formula:
        # Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
        # Subtracting mean ensures identifiability (unique V and A)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
