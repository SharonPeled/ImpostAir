import lightning as pl
from typing import Dict, Any, Optional
import torch 
from datetime import datetime

from toto.inference.forecaster import TotoForecaster

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

        # ------------------------------------------------------------------
        # Predict stage: autoregressive forecasting instead of teacher forcing
        # ------------------------------------------------------------------

        if stage == 'predict'and _type == 'autoregressive2':
            patch_size = self.toto.model.patch_embed.stride
            prediction_length = time_steps - patch_size  # same horizon as your loss uses

            # Build MaskedTimeseries for Toto
            # ts: [B, V, T]
            # ts_mask: [B, T] True where NaN/pad  -> need True where VALID for Toto
            padding_mask = (~toto_mask).bool()  # [B, V, T], True = valid

            # timestamps: [B, T] in ms -> seconds
            ts_seconds = (batch["timestamps"] / 1000.0).to(ts.device)  # [B, T]
            timestamp_seconds = ts_seconds.unsqueeze(1).expand(-1, n_input_features, -1)  # [B, V, T]

            # Estimate constant step in seconds per series
            dt = ts_seconds[:, 1:] - ts_seconds[:, :-1]  # [B, T-1]
            dt_med = dt.median(dim=1).values.clamp(min=1.0)  # [B]
            time_interval_seconds = dt_med.unsqueeze(1).expand(-1, n_input_features).to(torch.int32)  # [B, V]

            id_mask = torch.zeros_like(ts, dtype=torch.int32)

            inputs = MaskedTimeseries(
                series=ts,
                padding_mask=padding_mask,
                id_mask=id_mask,
                timestamp_seconds=timestamp_seconds.to(torch.int64),
                time_interval_seconds=time_interval_seconds,
            )

            forecaster = TotoForecaster(self.toto.model)

            with torch.no_grad():
                forecast = forecaster.forecast(
                    inputs,
                    prediction_length=prediction_length,
                    num_samples=None,          # or an int if you want samples
                    samples_per_batch=prediction_length,  # or any reasonable number
                    use_kv_cache=True,
                )

            # forecast.mean: [B, V, prediction_length] in ORIGINAL scale
            forecasts = forecast.mean
            targets = ts[:, :, patch_size:]  # [B, V, prediction_length]
            out_mask = toto_mask[:, :, patch_size:]  # [B, V, prediction_length]

            # Select only your output features
            forecasts = forecasts[:, target_idx, :]
            targets = targets[:, target_idx, :]
            out_mask = out_mask[:, target_idx, :]

            # MSE over valid points
            valid = ~out_mask
            sq_error = (targets - forecasts).pow(2)
            total_loss = (sq_error[valid]).mean() if valid.any() else torch.tensor(0.0, device=ts.device)

            # Per-sample loss
            valid_f = valid.float()
            loss_per_sample = (sq_error * valid_f).sum(dim=(1, 2)) / valid_f.sum(dim=(1, 2)).clamp(min=1.0)

            metrics = compute_batch_metrics(
                targets,
                forecasts,
                stage,
                out_mask,
                self.config["logging"]["forcasting_metrics"],
            )
            metrics[f"{stage}_loss"] = total_loss.detach().cpu()
            metrics[f"{stage}_NNL_loss"] = torch.tensor(0.0, device=ts.device).detach().cpu()
            metrics[f"{stage}_robust_loss"] = torch.tensor(0.0, device=ts.device).detach().cpu()

            return total_loss, metrics, loss_per_sample, y_track_is_anomaly


        if stage == 'predict' and _type == 'autoregressive':
            patch_size = self.toto.model.patch_embed.stride
            # Number of prediction steps matches the teacher-forcing setup
            pred_steps = time_steps - patch_size

            # We will iterate over the prediction horizon and at each step:
            # 1. Run the model on the full (possibly partially predicted) sequence
            # 2. Take the one-step-ahead forecast for this step
            # 3. Feed that forecast back into the sequence (autoregressive rollout)
            #
            # This keeps the training loss unchanged (teacher forcing) but
            # uses autoregressive forecasts during the "predict" stage.

            # Working copies so we can overwrite future targets with predictions
            ts_working = ts.clone()
            ts_mask_working = ts_mask.clone()

            # Storage for autoregressive forecasts with the same shape as in TotoLoss
            forecasts_ar = torch.zeros(
                batch_size,
                len(target_idx),
                pred_steps,
                device=ts.device,
                dtype=ts.dtype,
            )

            # Precompute the original targets and mask (do not mutate these)
            targets = ts[:, target_idx, patch_size:].clone()
            out_mask = toto_mask[:, target_idx, patch_size:].clone()

            with torch.no_grad():
                for step in range(pred_steps):
                    # Build mask for current working sequence
                    # [batch, time_steps] -> [batch, variate, time_steps]
                    step_toto_mask = ts_mask_working.unsqueeze(1).expand(-1, n_input_features, -1).bool()

                    toto_output = self.forward(
                        ts_working,
                        step_toto_mask,
                        timestamp=timestamp,
                        callsigns=callsigns,
                    )

                    # Convert base distribution back to the original data scale,
                    # mirroring TotoLoss (which uses TotoForecaster.create_affine_transformed).
                    base_distr = toto_output.distribution
                    loc = toto_output.loc
                    scale = toto_output.scale
                    distr = TotoForecaster.create_affine_transformed(base_distr, loc, scale)
                    full_forecasts = distr.mean  # [batch, variate, time_steps] in original scale

                    # One-step-ahead forecast for the relevant outputs at this step
                    step_forecast = full_forecasts[:, target_idx, step]  # [batch, n_targets]
                    forecasts_ar[:, :, step] = step_forecast

                    # Feed prediction back into the sequence at time (patch_size + step)
                    ts_working[:, target_idx, patch_size + step] = step_forecast
                    # Mark these positions as valid (not NaN / not padding)
                    ts_mask_working[:, patch_size + step] = False

            # Compute per-sample MSE over valid positions only
            valid_mask = ~out_mask  # True where we should evaluate
            sq_error = (targets - forecasts_ar).pow(2)

            # Total loss over all valid elements
            if valid_mask.any():
                total_loss = (sq_error * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
            else:
                total_loss = torch.tensor(0.0, device=ts.device, dtype=ts.dtype)

            # Per-sample loss (shape [batch_size])
            valid_mask_float = valid_mask.float()
            loss_per_sample = (sq_error * valid_mask_float).sum(dim=(1, 2)) / valid_mask_float.sum(dim=(1, 2)).clamp(min=1.0)

            # Metrics on autoregressive forecasts
            metrics = compute_batch_metrics(
                targets,
                forecasts_ar,
                stage,
                out_mask,
                self.config["logging"]['forcasting_metrics'],
            )

            metrics[f"{stage}_loss"] = total_loss.detach().cpu()

            # For predict stage we don't have NNL / robust decomposition; fill with zeros for logging consistency
            metrics[f"{stage}_NNL_loss"] = torch.tensor(0.0, device=ts.device).detach().cpu()
            metrics[f"{stage}_robust_loss"] = torch.tensor(0.0, device=ts.device).detach().cpu()
            
            nan_mask_shofted = out_mask.any(1).unsqueeze(1)
            timestamp_shifted = batch['timestamps'][:, :-self.toto_loss.patch_size].unsqueeze(1)

            return self..create_df_pred(timestamp_shifted, nan_mask_shofted, targets, forcasts, callsigns, paths)

        # ------------------------------------------------------------------
        # Train / val / test: standard teacher-forcing loss (unchanged)
        # ------------------------------------------------------------------
        toto_output = self.forward(ts, toto_mask, timestamp=timestamp, callsigns=callsigns)

        loss, loss_components, forcasts, targets, out_mask, loss_per_sample = self.toto_loss(
            toto_output, 
            inputs=ts, 
            padding_mask=toto_mask,
            target_idx=target_idx)

        if stage == 'predict' and self.pred_type == 'teacher_forcing':
            nan_mask_shofted = out_mask.any(1).unsqueeze(1)
            timestamp_shifted = batch['timestamps'][:, :-self.toto_loss.patch_size].unsqueeze(1)
            return self.create_df_pred(timestamp_shifted, nan_mask_shofted, targets, forcasts, callsigns, paths)

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


    def create_df_pred(self, timestamp, nan_mask, targets, forecasts, log_probs_per_step, callsigns, paths):

    data = torch. cat([
    timestamp, nan_mask, targets, forecasts,
    # log probs_ per _step
    ], dim=1).detach().cpu()
    cols = ['timestamp', 'is masked' ] \
    + [short_column_name(c, prefix='target_') for c in self.config["data"] ["output_features"]] \
    + [short_column_name(c, prefix='forecast_') for e in self.config["data"] ["output_features"]] \
    + [short_ column_namele, prefix='log_ prob_' for c in self.config["data"] ["output_features"]]
    # data - Shape: (batch, variates, time_steps)
    B = data.size(0)
    ts_list - []
    for i in range(B):
        data_single - data[1]
        print(len(cols), data_single.shape)
        df_ts = pd.DataFrame(data_single.T, columns=cols) 
        df_ts[' callsign'] = callsigns[i]
        df_ts['path'] = paths[i]
        ts_list.append(df_ts)
        df_cat = pd.concat(ts_list, ignore_index-True)
    return df_cat





    
    