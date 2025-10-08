import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Generator, Tuple
import math
import torch.nn.functional as F
from datetime import datetime 

from src.models.BaseNextPatchForecaster import BaseNextPatchForecaster
from src.models.positional_encoding import PositionalEncoding
from src.models.patch_embedding import PatchEmbedding
from src.models.CallsignEmbedding import CallsignEmbedding
from src.models.MultiScaleTimeEmbedding import MultiScaleTimeEmbedding


class PatchTransformerForecaster(BaseNextPatchForecaster):
    """
    Patch-based Decoder-Only Transformer for Next Patch Prediction.
    
    This model segments multivariate time series into patches and uses
    a transformer decoder to predict the next patch given context patches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(PatchTransformerForecaster, self).__init__(config)
        
        # Model architecture parameters
        self.patch_len = config['model']['patch_transformer_params']['patch_len']
        self.d_model = config['model']['patch_transformer_params']['d_model']
        self.n_heads = config['model']['patch_transformer_params']['n_heads']
        self.n_layers = config['model']['patch_transformer_params']['n_layers']
        self.d_ff = config['model']['patch_transformer_params']['d_ff']
        self.dropout = config['model']['patch_transformer_params']['dropout']
        self.activation = config['model']['patch_transformer_params']['activation']
        self.n_input_features = config['model']['patch_transformer_params']['n_input_features']
        self.n_output_features = config['model']['patch_transformer_params']['n_output_features']
        self.max_num_patches = config['model']['patch_transformer_params'].get('max_n_patches')
        self.pos_encoding_type = config['model']['patch_transformer_params'].get('pos_encoding_type')
        self.callsign_vocab_size = config['model']['patch_transformer_params']['callsign_vocab_size']
        self.time_embedding_scales = config['model']['patch_transformer_params']['time_embedding_scales']
        self.time_embedding_ref = datetime.strptime(config['model']['patch_transformer_params']['time_embedding_ref'], '%d/%m/%Y %H:%M:%S').timestamp()

        # Build model
        self._build_model()
        
        print(f"✓ Model initialized PatchTransformerForecaster.")
    
    def _build_model(self):
        """Build the patch-based transformer architecture."""

        self.callsign_embedding = CallsignEmbedding(
            vocab_size=self.callsign_vocab_size,
            d_model=self.d_model
        )

        self.time_embedding = MultiScaleTimeEmbedding(
            d_model=self.d_model,
            ref_timestamp=self.time_embedding_ref,
            scales=self.time_embedding_scales
        )
        
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
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the patch transformer.

        Args:
            batch (Dict[str, torch.Tensor]): 
                - 'ts': Tensor of shape [batch_size, num_patches, patch_length, n_input_features], 
                  the input sequence of patches.
                - 'nan_mask': Tensor of shape [batch_size, num_patches], 
                  boolean mask (True where value is NaN/imputed/pad).
                - 'callsign_idx': Tensor of shape [batch_size], 
                  integer index for callsign embedding.

        Returns:
            y_pred_all (torch.Tensor): 
                Tensor of shape [batch_size, num_patches, patch_length, n_output_features], 
                the predicted next patch for each input patch.
        """

        context_patches = batch['ts']
        context_patches_mask = batch['nan_mask']
        context_patches_callsign = batch['callsign_idx']
        timestamps = batch['timestamps']
        assert not context_patches.isnan().any(), "Context contains NaN"

        context_patches = context_patches

        B, N, L, C = context_patches.shape

        # Patch embedding + positional encoding
        x = self.patch_embedding(context_patches)  # [batch, num_patches, d_model]
        x = self.pos_encoding(x)  # [batch, num_patches, d_model]

        callsign_token = self.callsign_embedding(context_patches_callsign)  # [batch, d_model]
        time_embedding_token = self.time_embedding(timestamps[:, 0] / 1000)  # taking only first timestamp and converting to seconds

        init_token = callsign_token + time_embedding_token  # fusing both tokens
        x = torch.cat([init_token.unsqueeze(1), x], dim=1)  # [batch, num_patches + 1, d_model]
        context_patches_mask = torch.cat([torch.zeros(B, 1, dtype=context_patches_mask.dtype, device=context_patches_mask.device), context_patches_mask], dim=1)  # [batch, num_patches + 1]
        N += 1

        h = self.transformer(
            x,
            mask=nn.Transformer.generate_square_subsequent_mask(N, x.device), 
            src_key_padding_mask=context_patches_mask.float())  # [B, N, D]

        h = self.out_norm(h)

        y_pred_all = self.out_proj(h).view(B, N, self.patch_len, self.n_output_features)[:, 1:, :, :]  # removing the init callsign token

        return y_pred_all
    


    
    