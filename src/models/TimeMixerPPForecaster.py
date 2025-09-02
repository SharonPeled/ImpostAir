from src.models.TimeMixerPPBackbone import TimeMixerPPBackbone
import torch.nn as nn
import torch
import torch.nn.functional as F

from src.models.BaseNextPatchForecaster import BaseNextPatchForecaster
from src.models.PatchTransformerForecaster import PatchTransformerForecaster


class TimeMixerPPForecaster(BaseNextPatchForecaster):
    def __init__(self, config: dict):
        super(TimeMixerPPForecaster, self).__init__(config, patch_len=config['model']['patch_transformer_params']['patch_len'])
        self.backbone = TimeMixerPPBackbone(
            n_features=config['model']['time_mixer_pp_params']['n_features'],
            n_layers=config['model']['time_mixer_pp_params']['n_layers'],
            d_model=config['model']['time_mixer_pp_params']['d_model'],
            d_ffn=config['model']['time_mixer_pp_params']['d_ffn'],
            n_heads=config['model']['time_mixer_pp_params']['n_heads'],
            dropout=config['model']['time_mixer_pp_params']['dropout'],
            top_k=config['model']['time_mixer_pp_params']['top_k'],
            n_kernels=config['model']['time_mixer_pp_params']['n_kernels'],
            channel_mixing=config['model']['time_mixer_pp_params']['channel_mixing'],
            channel_independence=config['model']['time_mixer_pp_params']['channel_independence'],
            downsampling_layers=config['model']['time_mixer_pp_params']['downsampling_layers'],
            downsampling_window=config['model']['time_mixer_pp_params']['downsampling_window'],
            downsampling_method=config['model']['time_mixer_pp_params']['downsampling_method'],
            use_future_temporal_feature=config['model']['time_mixer_pp_params']['use_future_temporal_feature'],
            use_norm=config['model']['time_mixer_pp_params'].get('use_norm', False),
            embed=config['model']['time_mixer_pp_params'].get('embed', 'fixed'),
            freq=config['model']['time_mixer_pp_params'].get('freq', 'h'),
            n_classes=config['model']['time_mixer_pp_params'].get('n_classes', None)
        )

        self.aggregator = PatchTransformerForecaster(config)
    
    def forward(self, context: torch.Tensor, context_mask: torch.Tensor = None, callsign_id: torch.Tensor = None) -> torch.Tensor:
        enc_out_list = self.backbone.encode(context, context_mask)
        enc_out = enc_out_list[0] # [batch_size, context_steps, d_model] taking only first resolution
        return self.aggregator.forward(enc_out, context_mask, callsign_id=callsign_id)