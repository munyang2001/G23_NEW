import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        """
        The residual block is used for accommodating the vanishing gradient problem.
        Structure: 3x3 conv -> Batch norm -> relu -> 3x3 conv -> Batch norm
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        # The skip connection data to prevent information loss.
        residual = x

        # Passing through the convolutional layers.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # We add the final output with residual to prevent losing information when the gradient vanishes.
        out += residual

        out = self.relu(out)

        return out


class HexResNet(nn.Module):
    def __init__(self, board_size=11, in_channels=3, num_blocks=4, width=128):
        """
        An Alpha-zero style network: stem -> residual network tower -> policy head + value head.
        """
        super(HexResNet, self).__init__()
        # Hyperparameters
        self.board_size = board_size
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.width = width

        # The stem (entryway) that expands the input channels from 4 to width
        self.conv_input = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=width,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU()
        )

        # Creates a list of blocks that build the ResNet tower.
        blocks = [ResidualBlock(width) for _ in range(num_blocks)]

        # Wrap them in Sequential so PyTorch treats them as one big layer.
        self.res_tower = nn.Sequential(*blocks)

        # The policy head (the actor), assigns probabilities for every position on the board.
        self.policy_head = nn.Sequential(
            # Compress 128 channels into 2 channels("summarising").
            nn.Conv2d(
                in_channels=width,
                out_channels=2,
                kernel_size=1,
                bias=False
            ),

            # Normalise the output.
            nn.BatchNorm2d(2),
            nn.ReLU(),

            # Flatten the (1, 2, 11, 11) into 2 * 11 * 11 = 242 numbers.
            nn.Flatten(),

            # Assign probabilities for 121 positions calculated from the weighted sum of 242 numbers.
            nn.Linear(in_features=(2*self.board_size*self.board_size), out_features=board_size**2)
        )

        # The value head (the critic) that assigns a single number between -1(loss) and +1(win) indicating the winning chance.
        self.value_head = nn.Sequential(
            # Compress 128 channels into 1.
            nn.Conv2d(
                in_channels=width,
                out_channels=1,
                kernel_size=1,
                bias=False
            ),

            # Normalise the output.
            nn.BatchNorm2d(1),
            nn.ReLU(),

            nn.Flatten(),

            # The hidden dense layer (The "reasoning" layer).
            nn.Linear(in_features=(self.board_size**2), out_features=64),
            nn.ReLU(),
            # The final output (scalar)
            nn.Linear(in_features=64, out_features=1),
            # Tanh Activation (Forces output to be between -1 and 1)
            nn.Tanh()
        )


    def forward(self, x):
        """
        x: (N, C, H, W)
        Return:
            policy_logits: (N, board_size**2)
            value: (N, 1)
        """
        # 1. Input Safety Checks (Keep these, they are good)
        # -------------------------------------------------
        assert x.dim() == 4, f"Expected 4D input (N, C, H, W), got {x.shape}"
        N, C, H, W = x.shape
        assert N == 121, f"Expected 121 input board positions, got {N}"
        assert H == self.board_size and W == self.board_size, f"Bad Board Size: {H}x{W}"
        assert C == self.in_channels, f"Bad Channel Count: {C}"

        # 2. The Stem (Entry)
        # -------------------
        x = self.conv_input(x)  # Expands to 128 channels

        # 3. The Body (ResNet Tower)
        # --------------------------
        x = self.res_tower(x)   # The deep thinking happens here

        # 4. The Heads (Split)
        # --------------------
        policy_logits = self.policy_head(x)  # Shape: (N, 121)
        value = self.value_head(x)           # Shape: (N, 1)

        return policy_logits, value


    def predict(self, board_tensor):
        """
        Helper for MCTS.
        Used during competition.
        Input: board_tensor (1, C, H, W)
        Returns: (policy_probs, value)
        """
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board_tensor)

            # Apply Softmax to policy logits to get probabilities (0 to 1)
            probs = F.softmax(pi, dim=1)

            # Return as numpy arrays for the MCTS engine to use
            # [0] is used to unwrap the batch dimension
            return probs.cpu().numpy()[0], v.item()
