import torch
import torch.nn as nn

class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()
        # 1. For the Value Head (Regression)
        self.mse_loss = nn.MSELoss()

        # 2. For the Policy Head (Classification)
        # We do NOT use CrossEntropyLoss directly because it expects 'Class Indices'.
        # We have 'Probabilities' (MCTS visits). So we handle the math manually.

    def forward(self, pred_policy_logits, pred_value, target_probs, target_values):
        """
        pred_policy_logits: The raw output from the Policy Head (before Softmax)
        pred_value:         The raw output from the Value Head (between -1 and 1)
        target_probs:       The 'True' probabilities from MCTS (N, 121)
        target_value:       The 'True' winner (+1 or -1) (N, 1)
        """
        # --- A. Calculate Value Loss ---
        # Simple distance between Prediction and Reality
        value_loss = self.mse_loss(pred_value, target_values)

        # --- B. Calculate Policy Loss ---
        # 1. Convert Logits to Log-Probabilities (more numerically stable than log(softmax))
        log_probs = torch.log_softmax(pred_policy_logits, dim=1)

        # 2. Multiply by Target Probabilities (Cross Entropy Formula)
        # Formula: - sum( target * log(prediction) )
        policy_loss = -torch.mean(torch.sum(target_probs * log_probs, dim=1))

        # --- C. Combine ---
        total_loss = policy_loss + value_loss

        return total_loss