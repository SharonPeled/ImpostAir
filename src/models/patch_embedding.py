import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class PatchEmbedding(nn.Module):
    """Patch embedding layer for multivariate time series."""

    def __init__(self, patch_len: int, num_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.num_features = num_features
        self.d_model = d_model

        # Linear projection from flattened patch to d_model
        patch_input_dim = patch_len * num_features
        self.projection = nn.Linear(patch_input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: Tensor of shape [batch_size, num_patches, patch_len, num_features]

        Returns:
            Tensor of shape [batch_size, num_patches, d_model]
        """
        B, N, L, C = patches.shape
        # Flatten the last two dimensions (patch_len, num_features) into one
        patches_flat = patches.view(B, N, L * C)
        embedded = self.projection(patches_flat)
        embedded = self.dropout(embedded)
        return embedded
