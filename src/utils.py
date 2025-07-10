"""Utility functions for time series forecasting (metrics, helpers, etc.)."""
import torch
import torch.nn.functional as F
import importlib


def get_class_from_path(class_path):
    module_path, class_name = class_path.rsplit('.', 1)
    Class = getattr(importlib.import_module(module_path), class_name)
    return Class


def compute_patch_metrics(predicted_patches: torch.Tensor, 
                         target_patches: torch.Tensor,
                         patch_len: int,
                         num_features: int) -> dict:
    """
    Compute comprehensive metrics for patch prediction.
    
    Args:
        predicted_patches: [num_patches, patch_len * num_features]
        target_patches: [num_patches, patch_len * num_features]
        patch_len: Length of each patch
        num_features: Number of features
    
    Returns:
        Dictionary containing:
            mse (float): Mean squared error across all patches
            mae (float): Mean absolute error across all patches 
            rmse (float): Root mean squared error across all patches
            per_feature_mse (list): MSE per feature, averaged over patches and timesteps
    """
    num_patches, patch_dim = predicted_patches.shape
    
    # Reshape the patches to [batch_size, num_patches, patch_len, num_features]
    predicted_reshaped = predicted_patches.view(num_patches, patch_len, num_features)
    target_reshaped = target_patches.view(num_patches, patch_len, num_features)
    
    # Overall metrics
    mse = F.mse_loss(predicted_patches.view(-1), target_patches.view(-1))  # MSE over all patches
    mae = F.l1_loss(predicted_patches.view(-1), target_patches.view(-1))   # MAE over all patches
    rmse = torch.sqrt(mse)
    
    # Per-feature MSE: Compute MSE for each feature, averaged over all patches and all batches
    per_feature_mse = F.mse_loss(predicted_reshaped, target_reshaped, reduction='none')  # [num_patches, patch_len, num_features]
    per_feature_mse = per_feature_mse.mean(dim=(0, 1))  # Average over patches and time steps for each feature
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'rmse': rmse.item(),
        'per_feature_mse': per_feature_mse.tolist(),
    }