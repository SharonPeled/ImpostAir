import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """
    Time2Vec embedding for a single scalar timestamp.

    Args:
        k (int): Output embedding dimension (1 linear + k-1 periodic).
        ref_timestamp (int): Reference timestamp for normalization (in seconds).
        scale (str): Time scale for normalization. One of: 'hours', 'days', 'weeks', 'months', 'years'.

    Given a timestamp tau, computes:
      t2v(tau) = [w0 * tau + b0, sin(w1 * tau + b1), ..., sin(w_{k-1} * tau + b_{k-1})]
    where:
      - The first dimension is a linear projection.
      - The remaining (k-1) dimensions are periodic (sine) projections.

    Output shape: (..., k)
    """
    VALID_SCALES = {"hours": 3600, "days": 86400, "weeks": 604800, 
                    "months": 2629746, "years": 31556952}

    def __init__(self, k: int, ref_timestamp: int, scale: str):
        super().__init__()
        assert k >= 2, "k must be >= 2 (linear and periodic dims)"
        assert scale in self.VALID_SCALES, f"Invalid scale: {scale}"
        self._scale = self.VALID_SCALES[scale]
        self.ref_timestamp = ref_timestamp
        # linear term
        self.w0 = nn.Parameter(torch.randn(1) * 0.1)
        self.b0 = nn.Parameter(torch.randn(1) * 0.1)
        # periodic bank
        self.W = nn.Parameter(torch.randn(k-1) * 0.1)
        self.B = nn.Parameter(torch.randn(k-1) * 0.1)


    def normalize_tau(self, tau: torch.Tensor) -> torch.Tensor:
        return (tau - self.ref_timestamp) / self._scale

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        tau: [B,] timestamp in seconds
        returns: [B, k]
        """
        tau = self.normalize_tau(tau)  # [B,]
        lin = self.w0 * tau + self.b0  # [B,]
        lin = lin.unsqueeze(-1)        # [B, 1]
        per = torch.sin(tau.unsqueeze(-1) * self.W + self.B)  # [B, k-1]
        return torch.cat([lin, per], dim=-1)  # [B, k]