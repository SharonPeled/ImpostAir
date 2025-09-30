import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from toto.inference.forecaster import TotoForecaster


class TotoLoss(nn.Module):
    """
    TOTO loss function combining negative log-likelihood and robust loss.
    
    Based on the TOTO paper: https://arxiv.org/pdf/2505.14766
    """
    
    def __init__(self, 
                 lambda_NNL: float = 0.5755,
                 alpha: float = 0.0, 
                 delta: float = 0.1010,
                 patch_size: int = 64):
        """
        Args:
            lambda_robust: Weight for robust loss component
            alpha: Robust loss parameter (0.0 for MSE-like behavior)
            delta: Robust loss parameter (0.1010 for MSE-like behavior)
            use_robust_loss: Whether to include robust loss component
        """
        super().__init__()
        self.lambda_NNL = lambda_NNL
        self.alpha = alpha
        self.delta = delta
        self.patch_size = patch_size
    
    def robust_loss(self, forcasts: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute robust loss component.
        """
        residuals = forcasts - targets  #  [batch_size, num_variates, time_steps]
        
        if self.alpha == 0.0:
            # When alpha=0, robust loss becomes similar to MSE
            # L_robust = 2 * delta^2 * (sqrt(1 + (r/delta)^2) - 1)
            loss_per_time_step = 2 * self.delta**2 * (torch.sqrt(1 + (residuals / self.delta)**2) - 1)
        else:
            # General robust loss formula
            loss_per_time_step = 2 * self.delta**2 * (
                (self.alpha + 1) / self.alpha * 
                ((1 + (residuals / self.delta)**2 / (2 * self.alpha))**(self.alpha / 2) - 1)
            )
        
        loss_per_sample = (loss_per_time_step * (~padding_mask).float()).sum(dim=(1, 2)) \
                  / (~padding_mask).sum(dim=(1, 2)).clamp(min=1)

        return loss_per_sample
    
    def forward(self, 
                toto_output, 
                inputs: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                target_idx: Optional[list] = None) -> torch.Tensor:
        """
        Compute TOTO loss.
        
        Args:
            toto_output: TotoOutput object with distribution, loc, scale
            inputs: inputs values (batch, variate, time_steps)
            padding_mask: True where value is NaN/imputed/pad  (batch, time_steps)
        
        Returns:
            Total loss
        """
        # Extract components from TOTO output
        distribution = toto_output.distribution
        loc = toto_output.loc
        scale = toto_output.scale
        distr = TotoForecaster.create_affine_transformed(distribution, loc, scale)
        
        # 1. Negative Log-Likelihood Loss
        # Compute log probability of targets under the predicted distribution
        log_probs = distr.log_prob(inputs)  # Shape: (batch, variate, time_steps)

        # If none, using all variates
        if target_idx is None:
            target_idx = [True] * inputs.shape[1]

        # shifted log_probs and targets
        # Toto also remove extreme values that can occur early in training
        targets = inputs[:, target_idx, self.patch_size:]
        padding_mask = padding_mask[:, target_idx, self.patch_size:]
        log_probs = log_probs[:, target_idx, :-self.patch_size]

        log_probs_per_sample = (log_probs * (~padding_mask).float()).sum(dim=(1, 2)) \
                        / (~padding_mask).sum(dim=(1, 2)).clamp(min=1)
        nll_loss_per_sample = -log_probs_per_sample
        
        # Get point predictions (mean of distribution)
        forcasts = distribution.mean  # Shape: (batch, variate, time_steps)        
        forcasts = forcasts[:, target_idx, :-self.patch_size]

        robust_loss_per_sample = self.robust_loss(forcasts, targets, padding_mask)
        
        # Combine losses
        loss_per_sample = self.lambda_NNL*nll_loss_per_sample + (1-self.lambda_NNL) * robust_loss_per_sample

        total_loss = loss_per_sample.mean()
        nll_loss = nll_loss_per_sample.mean()
        robust_loss = robust_loss_per_sample.mean()

        return total_loss, [nll_loss, robust_loss], forcasts, targets, padding_mask, loss_per_sample
