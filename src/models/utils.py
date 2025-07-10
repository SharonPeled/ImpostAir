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
        num_patches = num_steps // patch_len  # TODO: currently discarding the last patch if it's not a full patch, consider padding it
        ts = ts[:, :num_patches * patch_len, :]  # Shape: [batch_size, num_patches * patch_len, num_features]
        patches = ts.reshape(batch_size, num_patches, patch_len * num_features)
        return patches


def teacher_forcing_pairs_generator(patches: torch.Tensor) -> Generator[Tuple[torch.Tensor, torch.Tensor], Any, Any]:
        """
        Generate a generator of context-target pairs for teacher forcing over patches.

        Args:
            patches: [batch_size, num_patches, patch_len * num_features]

        Returns: 
            Generator of context-target pairs for teacher forcing.
        """
        _, num_patches, _ = patches.shape
        for i in range(num_patches - 1):
            context_patches = patches[:, :i+1, :]
            target_patch = patches[:, i + 1, :]
            yield context_patches, target_patch