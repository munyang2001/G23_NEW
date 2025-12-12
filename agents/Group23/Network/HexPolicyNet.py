import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        """
        Standard ResNet Block: Conv -> BN -> ReLU -> Conv -> BN -> (+) -> ReLU
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class HexResNet(nn.Module):
    def __init__(self, board_size=11, in_channels=3, num_blocks=4, width=128):
        super(HexResNet, self).__init__()
        self.board_size = board_size
        self.in_channels = in_channels

        # 1. The Stem
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU()
        )

        # 2. The Tower
        self.res_tower = nn.Sequential(*[ResidualBlock(width) for _ in range(num_blocks)])

        # 3. Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(width, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size ** 2)
        )

        # 4. Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(width, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size ** 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Input Validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (Batch, Channel, Height, Width), got {x.shape}")

        # Forward Pass
        x = self.conv_input(x)
        x = self.res_tower(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

    def predict(self, board_tensor):
        """
        Robust inference helper.
        Automatically handles device placement to prevent 'Expected object of device...' errors.
        """
        self.eval()

        # CRITICAL FIX: Ensure input is on the same device as the model
        # We check the device of the first parameter in the model
        device = next(self.parameters()).device

        # If input is numpy or wrong device, move it
        if isinstance(board_tensor, torch.Tensor):
            board_tensor = board_tensor.to(device)
        else:
            # Fallback for numpy inputs
            board_tensor = torch.tensor(board_tensor, dtype=torch.float32, device=device)

        with torch.no_grad():
            pi, v = self.forward(board_tensor)

            # Softmax for probabilities
            probs = F.softmax(pi, dim=1)

            # Return numpy (CPU)
            return probs.cpu().numpy()[0], v.item()