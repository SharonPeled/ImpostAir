import lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any
from src.utils import compute_metrics


class BaseNextPatchForecaster(pl.LightningModule):
    """Base class for time series forecasting stagels."""
        
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.save_hyperparameters(config)

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
        ts_mask = batch["nan_mask"]
        columns = [col[0] for col in batch["columns"]]  # the column names are batched 
        target_idx = [columns.index(c) for c in self.config["data"]["output_features"]]

        y_pred = self.forward(ts, ts_mask)

        # shifting forcasts and y_true 
        y_pred = y_pred[:, :-1, :, :]  # filtering the last patch as we dont have its corroposing y_true
        y_true = ts[:, 1:, :, target_idx]  # shifting y_true by one patch to align with the forcasts
        y_mask = ts_mask[:, 1:]

        loss = self.loss(y_true, y_pred, y_mask)

        metrics = compute_metrics(
            y_true, y_pred, stage, y_mask, self.config["logging"][stage]
        )
        
        metrics[f"{stage}_loss"] = loss
        return loss, metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, metrics = self._process_batch(batch, stage="train")
        for name, val in metrics.items():
            self.log(name, val, on_step=True, on_epoch=True, prog_bar=name.endswith("loss"), sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, metrics = self._process_batch(batch, stage="val")
        for name, val in metrics.items():
            self.log(name, val, on_step=False, on_epoch=True, prog_bar=name.endswith("loss"), sync_dist=True)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, metrics = self._process_batch(batch, stage="test")
        for name, val in metrics.items():
            self.log(name, val, on_step=False, on_epoch=True, prog_bar=name.endswith("loss"), sync_dist=True)
        return loss

    def loss(self, y_true, y_pred, mask):
        se = (y_true - y_pred).pow(2).sum(dim=(2, 3))
        return se[mask].mean()
