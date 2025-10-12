"""Utility functions for time series forecasting (metrics, helpers, etc.)."""
import torch
import torch.nn.functional as F
import importlib
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score

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


def compute_batch_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    stage: str,
    mask: torch.Tensor = None,
    metric_list = None
) -> dict:
    """
    Compute selected metrics for patch prediction.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape [batch_size, num_steps, num_features].
        y_pred (torch.Tensor): Predicted tensor of shape [batch_size, num_steps, num_features].
        mask (torch.Tensor, optional): Boolean mask of shape [batch_size, num_steps] (True where value is NaN/imputed/pad).
        stage: train/test/val
        metric_list (list): List of metric names to compute. Supported: 'mse', 'mae', 'rmse'.

    Returns:
        dict: Dictionary containing the computed metrics.
    """
    y_true = y_true[~mask]
    y_pred = y_pred[~mask]

    results = {}
    for metric in metric_list:
        if metric == 'mse':
            mse = F.mse_loss(y_pred.view(-1), y_true.view(-1))
            results[f'{stage}_mse'] = mse.item()
        elif metric == 'mae':
            mae = F.l1_loss(y_pred.view(-1), y_true.view(-1))
            results[f'{stage}_mae'] = mae.item()
        elif metric == 'rmse':
            mse = F.mse_loss(y_pred.view(-1), y_true.view(-1))
            rmse = torch.sqrt(mse)
            results[f'{stage}_rmse'] = rmse.item()
        else:
            results[f'{stage}_{metric}'] = float('nan')

    return results


def compute_track_anomaly_metrics(
    y_true_track_is_anomaly: torch.Tensor,
    y_pred_track_is_anomaly: torch.Tensor,
    stage: str,
    metric_list = None,
    suffix = '_epoch'
) -> dict:
    """
    Compute selected metrics for patch prediction.

    Args:
        y_true_track_is_anomaly (np.array): [num_samples].
        y_pred_track_is_anomaly (np.array): [num_samples].
        stage: train/test/val
        metric_list (list): List of metric names to compute. Supported: 'mse', 'mae', 'rmse'.

    Returns:
        dict: Dictionary containing the computed metrics.
    """
    results = {}
    if np.isnan(y_true_track_is_anomaly).all():
        for metric in metric_list:
            results[f'{stage}_{metric}{suffix}'] = float('nan')
        return results

    for metric in metric_list:
        if metric == 'track_AD_accuracy':
            results[f'{stage}_track_AD_accuracy{suffix}'] = accuracy_score(
                y_true=y_true_track_is_anomaly,
                y_pred=y_pred_track_is_anomaly
                )
        else:
            results[f'{stage}_{metric}{suffix}'] = float('nan')

    return results