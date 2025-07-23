import torch
from typing import Generator, Tuple, Any


def divide_ts_into_patches(ts: torch.Tensor, patch_len: int) -> torch.Tensor: 
        """
        Divide a time series into patches.

        Args:
            ts: [batch_size, num_steps, num_features]

        Returns:
            patches: [batch_size, num_patches, patch_len * num_features]
        """
        batch_size, num_steps, num_features = ts.shape
        assert num_steps % patch_len == 0, "num_steps must be divisible by patch_len"
        num_patches = num_steps // patch_len 
        ts = ts[:, :num_patches * patch_len, :]  # Shape: [batch_size, num_patches * patch_len, num_features]
        patches = ts.reshape(batch_size, num_patches, patch_len * num_features)
        return patches


def teacher_forcing_pairs_generator(
    ts: torch.Tensor, patch_len: int, ts_mask: torch.Tensor = None
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Any, Any]:
    """
    Generate a generator of context-target pairs for teacher forcing over the raw time series.

    Args:
        ts: [batch_size, num_steps, num_features]
        patch_len: int, length of each patch
        ts_mask: [batch_size, num_steps] boolean mask (True where value is NaN/imputed/pad)
    Yields:
        context: [batch_size, context_steps, num_features]
        target_patch: [batch_size, patch_len, num_features]
        context_mask: [batch_size, context_steps] boolean mask (True where value is NaN/imputed/pad)
        target_patch_mask: [batch_size, patch_len] boolean mask (True where value is NaN/imputed/pad)
    """
    batch_size, num_steps, num_features = ts.shape
    num_patches = num_steps // patch_len  # Note: it assumes num_steps is divisible by patch_len, otherwise it will discard the last incomplete patch 
    for i in range(num_patches - 1):
        context_end = (i + 1) * patch_len
        target_start = context_end
        target_end = target_start + patch_len
        context = ts[:, :context_end, :]  # [batch_size, context_steps, num_features]
        target_patch = ts[:, target_start:target_end, :]  # [batch_size, patch_len, num_features]
        if ts_mask is None:
            yield context, target_patch, None, None
        else:
            context_mask = ts_mask[:, :context_end]  # [batch_size, context_steps]
            target_patch_mask = ts_mask[:, target_start:target_end]  # [batch_size, patch_len]
            yield context, target_patch, context_mask, target_patch_mask