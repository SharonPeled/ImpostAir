import lightning as pl
from typing import Dict, Any, Optional
import torch 

from src.models.BaseNextPatchForecaster import BaseNextPatchForecaster
from src.models.TotoBackboneWrapper import TotoBackboneWrapper
from src.utils import compute_metrics
from src.objects.TotoLoss import TotoLoss

class TotoWrapper(BaseNextPatchForecaster):
    """Wrapper for Toto model."""

    def __init__(self, config: Dict[str, Any]):
        super(TotoWrapper, self).__init__(config)

        # Build model
        self.toto = TotoBackboneWrapper.from_pretrained(
            self.config['model']['toto_params']['from_pretrained'],
            strict=True
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

        ts_mask = batch["nan_mask"]  # batch, timestemps
        if ts_mask is not None:
            # ts_mask: [batch, time_steps] -> [batch, 1, time_steps] -> [batch, variate, time_steps]
            ts_mask = ts_mask.unsqueeze(1).expand(-1, n_input_features, -1).bool()
        else:
            ts_mask = torch.zeros((batch_size, n_input_features, time_steps), dtype=torch.bool, device=ts.device)

        columns = [col[0] for col in batch["columns"]]  # the column names are batched 
        target_idx = [columns.index(c) for c in self.config["data"]["output_features"]]

        toto_output = self.forward(ts, ts_mask)

        loss, loss_components, forcasts, targets, out_mask = self.toto_loss(
            toto_output, 
            inputs=ts, 
            padding_mask=ts_mask,
            target_idx=target_idx)

        metrics = compute_metrics(
            targets, forcasts, stage, out_mask, self.config["logging"][stage]
        )
        
        metrics[f"{stage}_loss"] = loss.detach().cpu()
        metrics[f"{stage}_NNL_loss"] = loss_components[0].detach().cpu()
        metrics[f"{stage}_robust_loss"] = loss_components[1].detach().cpu()
        return loss, metrics    

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor = None,
        id_mask: torch.Tensor = None,
        scaling_prefix_length: Optional[int] = None,
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
            id_mask = torch.ones((batch_size, n_input_features, time_steps), dtype=context.dtype, device=context.device)

        context_mask = ~context_mask  # toto assumes opposite mask 

        return self.toto(
            context,
            context_mask,
            id_mask,
            kv_cache=None,  # currently not supported
            scaling_prefix_length=scaling_prefix_length
        )
    





    
    