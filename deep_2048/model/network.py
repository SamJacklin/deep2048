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

        # Main path: three 3×3 layers capture local interactions and emergent
        # global structure across the 4×4 board.
        self.conv_main = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Row path: 1×4 kernels span an entire row in one step.
        # Directly relevant to LEFT/RIGHT merges and monotone row structure.
        self.conv_row = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(1, board_size)),
            nn.ReLU(),
        )

        # Column path: 4×1 kernels span an entire column in one step.
        # Directly relevant to UP/DOWN merges and monotone column structure.
        self.conv_col = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(board_size, 1)),
            nn.ReLU(),
        )

        # main: 128 × board_size × board_size
        # row:  64  × board_size × 1  →  64 × board_size
        # col:  64  × 1 × board_size  →  64 × board_size
        flat_dim = (
            128 * board_size * board_size
            + 64 * board_size
            + 64 * board_size
        )

        self.feature_fc = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
        )

        # Dueling streams — unchanged shape so save/load inference still works.
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
        x = x.view(-1, self.input_channels, self.board_size, self.board_size)

        main_feat = self.conv_main(x).flatten(1)
        row_feat = self.conv_row(x).flatten(1)
        col_feat = self.conv_col(x).flatten(1)

        features = self.feature_fc(
            torch.cat([main_feat, row_feat, col_feat], dim=1)
        )
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
