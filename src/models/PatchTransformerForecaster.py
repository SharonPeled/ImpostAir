import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Generator, Tuple
import math
import torch.nn.functional as F

from src.models.BaseNextPatchForecaster import BaseNextPatchForecaster
from src.models.positional_encoding import PositionalEncoding
from src.models.patch_embedding import PatchEmbedding
from src.models.utils import divide_ts_into_patches


class PatchTransformerForecaster(BaseNextPatchForecaster):
    """
    Patch-based Decoder-Only Transformer for Next Patch Prediction.
    
    This model segments multivariate time series into patches and uses
    a transformer decoder to predict the next patch given context patches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(PatchTransformerForecaster, self).__init__(config, patch_len=config['model']['patch_transformer_params']['patch_len'])
        
        # Model architecture parameters
        self.d_model = config['model']['patch_transformer_params']['d_model']
        self.n_heads = config['model']['patch_transformer_params']['n_heads']
        self.n_layers = config['model']['patch_transformer_params']['n_layers']
        self.d_ff = config['model']['patch_transformer_params']['d_ff']
        self.dropout = config['model']['patch_transformer_params']['dropout']
        self.activation = config['model']['patch_transformer_params']['activation']
        self.n_input_features = config['model']['patch_transformer_params']['n_input_features']
        self.n_output_features = config['model']['patch_transformer_params']['n_output_features']
        self.max_num_patches = config['model']['patch_transformer_params'].get('max_n_patches')
        self.context_length = config['model']['patch_transformer_params']['context_length']
        self.pos_encoding_type = config['model']['patch_transformer_params'].get('pos_encoding_type')
        self.patch_nan_tolerance_percentage = config['model']['patch_transformer_params']['patch_nan_tolerance_percentage']
        
        # Build model
        self._build_model()
        
        print(f"✓ Model initialized PatchTransformerForecaster.")
    
    def _build_model(self):
        """Build the patch-based transformer architecture."""
        
        # Patch embedding layer
        # TODO: create better embedder 
        self.patch_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            num_features=self.n_input_features,
            d_model=self.d_model,
            dropout=self.dropout
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_num_patches=self.max_num_patches,
            encoding_type=self.pos_encoding_type
        )
        
        # Transformer decoder layers (using PyTorch built-in)
        # Using TransformerEncoder with a casual mask to create a decoder-only model
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.n_layers)

        # Layer normalization before head
        self.out_norm = nn.LayerNorm(self.d_model)

        # Next patch prediction head
        patch_output_dim = self.patch_len * self.n_output_features
        self.out_proj = nn.Linear(self.d_model, patch_output_dim)  
    
    def forward(self, context_patches: torch.Tensor, context_patches_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the patch transformer.

        Args:
            context_patches: [batch_size, num_patches, patch_length, n_input_features]
            context_mask: [batch_size, num_patches] boolean mask (True where value is NaN/imputed/pad)

        Returns:
            next_patch_pred: [batch_size, num_patches-1, n_output_features]
        """
        assert not context_patches.isnan().any(), "Context contains NaN"

        context_patches = context_patches

        B, N, L, C = context_patches.shape

        # Patch embedding + positional encoding
        x = self.patch_embedding(context_patches)  # [batch, num_patches, d_model]
        x = self.pos_encoding(x)  # [batch, num_patches, d_model]

        h = self.transformer(
            x,
            mask=nn.Transformer.generate_square_subsequent_mask(N, x.device), 
            src_key_padding_mask=context_patches_mask.float())  # [B, N, D]

        h = self.out_norm(h)

        # predict all next patches in parallel (teacher forcing, shifted)
        # Use states at 0..T-2 to predict labels at 1..T-1, and mask pads in the loss.
        h_shift = h[:, :-1, :]

        y_pred_all = self.out_proj(h_shift).view(B, (N-1), self.patch_len, self.n_output_features)

        return y_pred_all
    
    def _get_key_padding_mask(self, context_patches: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        """
        Get key padding mask for the transformer decoder.
        context_mask: [batch_size, context_steps] boolean mask (True where value is NaN/imputed/pad)
        If context_mask is None, return mask with all False (no padding).
        """
        if context_mask is None:
            # No mask provided: return all False
            batch_size, num_patches, _ = context_patches.shape
            key_padding_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=context_patches.device)
        else:
            # context_mask: [batch_size, context_steps]
            mask_patches = divide_ts_into_patches(context_mask.unsqueeze(-1), self.patch_len)  # [batch, num_patches, patch_len]
            nan_counts = mask_patches.sum(dim=-1).float()  # [batch_size, num_patches]
            patch_nan_percentage = nan_counts / self.patch_len  # [batch_size, num_patches]
            key_padding_mask = patch_nan_percentage > self.patch_nan_tolerance_percentage  # [batch_size, num_patches], True = ignore
        
        return key_padding_mask


    
    