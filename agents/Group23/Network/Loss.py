import torch
import torch.nn as nn


class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_policy_logits, pred_value, target_probs, target_values):
        """
        Returns: (total_loss, policy_loss, value_loss)
        """
        # --- A. Calculate Value Loss ---
        # Fix shape mismatch: Flatten (64, 1) -> (64) to match target
        v_pred = pred_value.view(-1)
        v_targ = target_values.view(-1)

        value_loss = self.mse_loss(v_pred, v_targ)

        # --- B. Calculate Policy Loss ---
        log_probs = torch.log_softmax(pred_policy_logits, dim=1)
        policy_loss = -torch.mean(torch.sum(target_probs * log_probs, dim=1))

        # --- C. Combine ---
        total_loss = policy_loss + value_loss

        # CRITICAL FIX: Return a tuple of 3 values so the training loop can unpack them
        return total_loss, policy_loss, value_loss