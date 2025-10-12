import lightning as pl
from typing import Dict, Any, Optional
import torch 
from datetime import datetime

from src.models.BaseNextPatchForecaster import BaseNextPatchForecaster
from src.models.TotoBackboneWrapper import TotoBackboneWrapper
from src.utils import compute_batch_metrics
from src.objects.TotoLoss import TotoLoss


class TotoWrapper(BaseNextPatchForecaster):
    """Wrapper for Toto model."""

    def __init__(self, config: Dict[str, Any]):
        super(TotoWrapper, self).__init__(config)

        # Build model
        self.toto = TotoBackboneWrapper.from_pretrained(
            self.config['model']['toto_params']['from_pretrained'],
            strict=True,
            callsign_vocab_size=config['model']['toto_params']['callsign_vocab_size'],
            time_embedding_scales=config['model']['toto_params']['time_embedding_scales'],
            time_embedding_ref=datetime.strptime(config['model']['toto_params']['time_embedding_ref'], '%d/%m/%Y %H:%M:%S').timestamp()
            )

        # Build loss object
        self.toto_loss = TotoLoss(
            lambda_NNL=config['model']['toto_params']['loss']['lambda_NNL'],
            alpha=config['model']['toto_params']['loss']['alpha'],
            delta=config['model']['toto_params']['loss']['delta'],
            patch_size=self.toto.model.patch_embed.stride
            )
        
        print(f"✓ Toto model initialized.")
    
    def _process_batch(self, batch: Dict[str, torch.Tensor], stage: str):
        ts = batch["ts"]
        ts = ts.permute(0, 2, 1).contiguous()  # batch, variant, timestemps
        batch_size, n_input_features, time_steps = ts.shape
        y_track_is_anomaly = batch["y_track_is_anomaly"]
        ts_mask = batch["nan_mask"]  # batch, timestemps - True where value is NaN/imputed/pad

        callsigns = batch['callsign_idx']
        timestamp = batch['timestamps'][:, 0] / 1000

        # toto_mask: [batch, time_steps] -> [batch, 1, time_steps] -> [batch, variate, time_steps]
        toto_mask = ts_mask.unsqueeze(1).expand(-1, n_input_features, -1).bool()
        
        columns = [col[0] for col in batch["columns"]]  # the column names are batched 
        target_idx = [columns.index(c) for c in self.config["data"]["output_features"]]

        toto_output = self.forward(ts, toto_mask, timestamp=timestamp, callsigns=callsigns)

        loss, loss_components, forcasts, targets, out_mask, loss_per_sample = self.toto_loss(
            toto_output, 
            inputs=ts, 
            padding_mask=toto_mask,
            target_idx=target_idx)

        metrics = compute_batch_metrics(
            targets, forcasts, stage, out_mask, self.config["logging"]['forcasting_metrics']
        )
        
        metrics[f"{stage}_loss"] = loss.detach().cpu()
        metrics[f"{stage}_NNL_loss"] = loss_components[0].detach().cpu()
        metrics[f"{stage}_robust_loss"] = loss_components[1].detach().cpu()

        return loss, metrics, loss_per_sample, y_track_is_anomaly  

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor = None,
        id_mask: torch.Tensor = None,
        scaling_prefix_length: Optional[int] = None,
        timestamp: Optional[torch.Tensor] = None,
        callsigns: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Toto model.

        Args:
            context: [batch_size, num_timesteps, n_input_features]
                Input tensor of patches, where n_input_features is treated as "variate".
            context_mask: [batch_size, num_timesteps]
                Boolean mask (True where value is NaN/imputed/pad).
            id_mask: Optional[Tensor], default None
                Optional id mask. If None, a mask of ones will be used.
            kv_cache: Optional, default None
                Optional KVCache for autoregressive decoding.
            scaling_prefix_length: Optional[int], default None
                Optional prefix length for scaling.

        Returns:
            TotoOutput: Output from the Toto model.
        """

        # We'll treat n_input_features as "variate"
        batch_size, n_input_features, time_steps = context.shape

        # id_mask: [batch, #variate, time_steps] (float)
        if id_mask is None:
            id_mask = torch.zeros((batch_size, n_input_features, time_steps), dtype=context.dtype, device=context.device)

        context_mask = ~context_mask  # toto assumes opposite mask, (False where value is NaN/imputed/pad)

        return self.toto(
            context,
            context_mask,
            id_mask,
            kv_cache=None,  # currently not supported
            scaling_prefix_length=scaling_prefix_length,
            timestamp=timestamp,
            callsigns=callsigns
        )
    





    
    