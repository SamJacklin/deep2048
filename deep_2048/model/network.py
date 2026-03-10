import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(
        self,
        input_channels: int = 16,
        board_size: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 4,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.board_size = board_size

        # Conv layers operate on (batch, input_channels, board_size, board_size).
        # kernel_size=3, padding=1 preserves spatial dimensions at each layer.
        conv_out_channels = 128
        conv_flat_dim = conv_out_channels * board_size * board_size

        self.feature = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_flat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept flat input (batch, input_channels * board_size * board_size)
        # and reshape to (batch, input_channels, board_size, board_size).
        x = x.view(-1, self.input_channels, self.board_size, self.board_size)
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
