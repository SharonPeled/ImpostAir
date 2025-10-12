import lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any
from src.utils import compute_batch_metrics, compute_track_anomaly_metrics
import numpy as np 


class BaseNextPatchForecaster(pl.LightningModule):
    """Base class for time series forecasting stagels."""
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.save_hyperparameters(config)
        self._epoch_artifacts = {}
        self.anomaly_percentile = config["model"]["anomaly_percentile"]
        self.anomaly_train_threshold = None 

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Cosine annealing scheduler
        # TODO: should be in config
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['max_epochs'],
            eta_min=self.learning_rate * 0.01  # Minimum LR is 1% of initial LR
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # TODO: should be in config
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _process_batch(self, batch: Dict[str, torch.Tensor], stage: str):
        ts = batch["ts"]
        ts_mask = batch["nan_mask"]  # True where value is NaN/imputed/pad
        y_track_is_anomaly = batch["y_track_is_anomaly"]
        columns = [col[0] for col in batch["columns"]]
        target_idx = [columns.index(c) for c in self.config["data"]["output_features"]]

        y_pred = self.forward(ts, ts_mask)

        # shifting forcasts and y_true 
        y_pred = y_pred[:, :-1, :, :]  # filtering the last patch as we dont have its corroposing y_true
        y_true = ts[:, 1:, :, target_idx]  # shifting y_true by one patch to align with the forcasts
        y_mask = ts_mask[:, 1:]

        loss, batch_losses = self.loss(y_true, y_pred, y_mask)

        metrics = compute_batch_metrics(
            y_true, y_pred, stage, y_mask, self.config["logging"]['forcasting_metrics']
        )
        
        metrics[f"{stage}_loss"] = loss
        return loss, metrics, batch_losses, y_track_is_anomaly
    
    def general_step(self, batch, batch_idx, stage):
        loss, metrics, batch_losses, y_track_is_anomaly = self._process_batch(batch, stage=stage)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self._epoch_artifacts[stage]['loss'].append(batch_losses.detach().cpu())
        self._epoch_artifacts[stage]['y_track_is_anomaly'].append(y_track_is_anomaly.detach().cpu())
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, stage='train')

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self.general_step(batch, batch_idx, stage='val')

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self.general_step(batch, batch_idx, stage='test')
    
    def on_epoch_start_general(self, stage):
        self._epoch_artifacts[stage] = {
            'loss': [],
            'y_track_is_anomaly': []
        }

    def on_train_epoch_start(self):
        self.on_epoch_start_general(stage='train')
    
    def on_validation_epoch_start(self):
        self.on_epoch_start_general(stage='val')
    
    def on_test_epoch_start(self):
        self.on_epoch_start_general(stage='test')

    def on_epoch_end_general(self, stage):
        if len(self._epoch_artifacts[stage]) == 0:
            return
        epoch_losses = torch.cat(self._epoch_artifacts[stage]['loss'], dim=0).numpy()   # shape [N_total]
        epoch_y_track_is_anomaly = torch.cat(self._epoch_artifacts[stage]['y_track_is_anomaly'], dim=0).numpy()   # shape [N_total]
        if stage == 'train': 
            self.anomaly_train_threshold = float(np.percentile(epoch_losses, self.anomaly_percentile))
            self.log("anomaly_train_threshold", self.anomaly_train_threshold, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        epoch_y_pred_track_is_anomaly = (epoch_losses > self.anomaly_train_threshold).astype(float)
        metrics = compute_track_anomaly_metrics(epoch_y_track_is_anomaly, epoch_y_pred_track_is_anomaly, stage, self.config["logging"]['anomaly_detection_metrics'])
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def on_train_epoch_end(self):
        self.on_epoch_end_general(stage='train')
    
    def on_val_epoch_end(self):
        self.on_epoch_end_general(stage='val')
    
    def on_test_epoch_end(self):
        self.on_epoch_end_general(stage='test')

    def loss(self, y_true, y_pred, mask):
        """
        mask: True are valids, False should be filtered.
        """
        se = (y_true - y_pred).pow(2).mean(dim=(2, 3))
        # Compute the mean loss over all valid (masked) elements (scalar)
        total_loss = se[~mask].mean()
        # Compute the mean loss per sample (shape [B]), averaging only over valid (masked) elements per sample
        # For each sample, sum the losses where mask is True, and divide by the number of valid elements per sample
        mask_float = (~mask).float()  # [B, N]
        per_sample_loss = (se * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        return total_loss, per_sample_loss

