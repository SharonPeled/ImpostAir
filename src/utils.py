"""Utility functions for time series forecasting (metrics, helpers, etc.)."""
import torch
import torch.nn.functional as F
import importlib
from torchvision import transforms


def compose_transforms(config):
    transforms_list = []
    for transform_name in config['transformations']:
        transform_config = config['transformations'][transform_name]
        transform_class = get_class_from_path(transform_config['class_path'])
        transforms_list.append(transform_class(**transform_config['params']))
    return transforms.Compose(transforms_list)


def get_class_from_path(class_path):
    module_path, class_name = class_path.rsplit('.', 1)
    Class = getattr(importlib.import_module(module_path), class_name)
    return Class


def compute_patch_metrics(predicted_ts: torch.Tensor, 
                         target_ts: torch.Tensor, mask_ts: torch.Tensor = None) -> dict:
    """
    Compute comprehensive metrics for patch prediction.

    Args:
        predicted_ts: [num_steps, num_features]
        target_ts: [num_steps, num_features]
        mask_ts: [num_steps, ] boolean mask (True where value is NaN/imputed/pad)
    Returns:
        Dictionary containing:
            mse (float): Mean squared error across all patches
            mae (float): Mean absolute error across all patches 
            rmse (float): Root mean squared error across all patches
            per_feature_mse (list): MSE per feature, averaged over patches and timesteps
    """
    if mask_ts is not None:
        valid_mask = ~mask_ts
        predicted_ts = predicted_ts[valid_mask]
        target_ts = target_ts[valid_mask]
    
    # Overall metrics
    mse = F.mse_loss(predicted_ts.view(-1), target_ts.view(-1))  
    mae = F.l1_loss(predicted_ts.view(-1), target_ts.view(-1)) 
    rmse = torch.sqrt(mse)

    # Per-feature MSE: Compute MSE for each feature, averaged over all timesteps
    per_feature_mse = F.mse_loss(predicted_ts, target_ts, reduction='none')  # [num_steps, num_features]
    per_feature_mse = per_feature_mse.mean(dim=0)  # Average over timesteps for each feature

    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'rmse': rmse.item(),
        'per_feature_mse': per_feature_mse.tolist(),
    }
