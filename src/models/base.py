import lightning as pl
from typing import Dict, Any
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.models.utils import divide_ts_into_patches, teacher_forcing_pairs_generator
from src.utils import compute_patch_metrics


class BaseNextPatchForecaster(pl.LightningModule):
    """Base class for time series forecasting models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']

        self.patch_len = config['model']['patch_len']
        self.stride = config['model']['stride']

        self.save_hyperparameters(config)
    
    def _init_weights(self):
        """Initialize model weights following best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
    
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
    
    def general_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, mode: str):
        loss_list = []
        predicted_patch_list = []
        target_patch_list = []
        patches = divide_ts_into_patches(batch['ts'], self.patch_len)

        if patches.shape[1] <= 1:  
            print(f"Skipping batch {batch_idx} with not enough time steps.")
            return None

        if torch.isnan(patches).any():
            # TODO: handle this case better
            print(f"Skipping batch {batch_idx} with nan values.")
            return None

        for context_patches, target_patch in teacher_forcing_pairs_generator(patches):
            predicted_patch = self.forward(context_patches)  

            loss = self.loss(predicted_patch, target_patch)        
            loss_list.append(loss)

            predicted_patch_list.append(predicted_patch)
            target_patch_list.append(target_patch)
        
        loss = sum(loss_list) / len(loss_list)

        # Compute metrics
        metrics = compute_patch_metrics(
            torch.cat(predicted_patch_list, dim=0), 
            torch.cat(target_patch_list, dim=0),
            self.patch_len
        )
        
        # Log metrics
        self.log_metric(f'{mode}_loss', loss)
        self.log_metric(f'{mode}_mse', metrics['mse'])
        self.log_metric(f'{mode}_mae', metrics['mae'])
        self.log_metric(f'{mode}_rmse', metrics['rmse'])

        return loss    

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        loss = self.general_step(batch, batch_idx, 'train')
        if loss is None:
            return None
        return {'loss': loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        loss = self.general_step(batch, batch_idx, 'val')
        if loss is None:
            return None
        return {'val_loss': loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        loss = self.general_step(batch, batch_idx, 'test')
        if loss is None:
            return None
        return {'test_loss': loss}
    
    def loss(self, predicted_patch, target_patch):
        return F.mse_loss(predicted_patch, target_patch)
    
    def log_metric(self, metric_name: str, value: float):
        self.logger.experiment.log_metric(self.logger.run_id, metric_name, value)

