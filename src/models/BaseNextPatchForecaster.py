import lightning as pl
from typing import Dict, Any
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.utils import compute_patch_metrics


class BaseNextPatchForecaster(pl.LightningModule):
    """Base class for time series forecasting models."""
        
    def __init__(self, config: Dict[str, Any], patch_len: int):
        super().__init__()
        self.config = config
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.patch_len = patch_len
        self.save_hyperparameters(config)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['max_epochs'],
            eta_min=self.learning_rate * 0.01  # Minimum LR is 1% of initial
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _process_batch(self, batch: Dict[str, torch.Tensor], batch_idx: int, mode: str):
        """ """

        ts = batch['ts']
        ts_mask = batch['nan_mask']

        columns = [col[0] for col in batch['columns']]
        columns_indexes = [columns.index(col) for col in self.config['data']['output_features']]

        y_next_token_pred = self.forward(ts, ts_mask)  #  [B, ]
        y_true = ts[:, :-1, :, columns_indexes]  # shifting and taking only target variables
        y_true_mask = ts_mask[:, :-1]

        loss = self.loss(y_true, y_next_token_pred, y_true_mask)        
        
        # Compute metrics
        metrics = compute_patch_metrics(y_true, y_next_token_pred, y_true_mask)
        
        # Log metrics
        self.log_metric(f'{mode}_loss', loss)
        self.log_metric(f'{mode}_mse', metrics['mse'])
        self.log_metric(f'{mode}_mae', metrics['mae'])
        self.log_metric(f'{mode}_rmse', metrics['rmse'])

        return loss
    
    def general_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, mode: str):
        loss = self._process_batch(batch, batch_idx, mode)
        if loss is None:
            return None
        return {'loss': loss}  

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        return self.general_step(batch, batch_idx, 'train')

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        return self.general_step(batch, batch_idx, 'val')

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        return self.general_step(batch, batch_idx, 'test')
    
    def loss(self, y_true, y_next_token_pred, mask_pred):
        se = (y_true - y_next_token_pred).pow(2).sum(dim=(2,3))   # [B, N-1]
        return se[mask_pred].mean()
    
    def log_metric(self, metric_name: str, value: float):
        self.logger.experiment.log_metric(self.logger.run_id, metric_name, value)
        self.log(metric_name, value, prog_bar=True)
  