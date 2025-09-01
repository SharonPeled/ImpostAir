import lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any
from src.utils import compute_metrics
import numpy as np


class BaseNextPatchForecaster(pl.LightningModule):
    """Base class for time series forecasting stagels."""
        
    def __init__(self, config: Dict[str, Any], patch_len: int):
        super().__init__()
        self.config = config
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.patch_len = patch_len
        self.save_hyperparameters(config)
        self.anomaly_threshold_train = 0.5
        self._epoch_train_losses = []

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
        y_detected = batch["y_detected"]
        columns = [col[0] for col in batch["columns"]]
        target_idx = [columns.index(c) for c in self.config["data"]["output_features"]]

        y_pred = self.forward(ts, ts_mask)
        y_true = ts[:, :-1, :, target_idx]
        y_detected = y_detected[:, :-1]
        y_mask = ts_mask[:, :-1]

        loss = self.loss(y_true, y_pred, y_mask)

        metrics = compute_metrics(
            y_true=y_true, y_pred=y_pred, stage=stage, mask=y_mask, y_detected=y_detected, threshold=self.anomaly_threshold_train, metric_list=self.config["logging"][stage]
        )
        metrics[f"{stage}_loss"] = loss
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._process_batch(batch, stage="train")
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self._epoch_train_losses.append(loss.detach())
        return {"loss": loss}
        
    def on_train_epoch_start(self):
        self._epoch_train_losses = []

    def on_train_epoch_end(self):
        if len(self._epoch_train_losses) == 0:
            return
        per_step = torch.stack(self._epoch_train_losses)
        all_losses = self.all_gather(per_step).reshape(-1).float().cpu().numpy()
        self.anomaly_threshold_train = float(np.percentile(all_losses, 99))
        self.log("anomaly_threshold", self.anomaly_threshold_train, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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

    def process_batch_pred(self, batch: Dict[str, torch.Tensor], stage: str):
        ts = batch["ts"]                # [B, T, F]
        ts_mask = batch["nan_mask"]     # [B, T]
        y_detected = batch.get("y_detected", None)
        columns = [col[0] for col in batch["columns"]]
        target_idx = [columns.index(c) for c in self.config["data"]["output_features"]]

        # forward
        y_pred = self.forward(ts, ts_mask)
        y_true = ts[:, :-1, :, target_idx]      # align with pred
        y_mask = ts_mask[:, :-1]

        # compute loss
        loss = self.loss(y_true, y_pred, y_mask)

        paths = batch.get("path", [None] * ts.shape[0])
        if isinstance(paths, str):
            paths = [paths]

        # squared error for exporting
        se = (y_true - y_pred).pow(2).sum(dim=(2, 3))  # [B, T]
        flags = (se > getattr(self, "anomaly_threshold", 0.5)).int()

         # metrics (including detection_accuracy via compute_metrics)
        metrics = compute_metrics(
            y_true, 
            y_pred, 
            stage="predict",
            y_detected=y_detected[:, :-1] if y_detected is not None else None,
            threshold=getattr(self, "anomaly_threshold", 0.5),
            y_detected_pred=flags.detach().cpu().tolist() if y_detected is not None else None,
            mask=y_mask,
            metric_list=self.config["logging"][stage]
        )

        out = {
            "path": list(paths),
            "patch_scores": se.detach().cpu().tolist(),
            "is_anomaly": flags.detach().cpu().tolist(),
        }

        return loss, metrics, out

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        """Predict API for Lightning: delegates to process_batch_pred and logs."""
        loss, metrics, out = self.process_batch_pred(batch, stage="test")
        # self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return out
