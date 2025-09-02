import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Generator, Tuple
import math
import torch.nn.functional as F

from src.models.BaseNextPatchForecaster import BaseNextPatchForecaster
from src.models.positional_encoding import PositionalEncoding
from src.models.patch_embedding import PatchEmbedding


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
        self.pos_encoding_type = config['model']['patch_transformer_params'].get('pos_encoding_type')
        self.callsign_num_buckets = config.get('data', {}).get('callsign_num_buckets', 4096)
        self.callsign_embed_dim = config['model']['patch_transformer_params'].get('callsign_embed_dim', 0)
        
        
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
        if self.callsign_embed_dim and self.callsign_embed_dim > 0:
            self.callsign_embedding = nn.Embedding(self.callsign_num_buckets, self.callsign_embed_dim)
            self.callsign_project = nn.Linear(self.callsign_embed_dim, self.d_model)
        else:
            self.callsign_embedding = None
            self.callsign_project = None
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
    
    def forward(self, context_patches: torch.Tensor, context_patches_mask: torch.Tensor = None, callsign_id: torch.Tensor = None) -> torch.Tensor:
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
        # Inject callsign embedding if available
        if self.callsign_embedding is not None and callsign_id is not None:
            # callsign_id expected shape [B]
            if callsign_id.dim() > 1:
                callsign_id = callsign_id.squeeze()
            cs_emb = self.callsign_embedding(callsign_id.long())               # [B, E]
            cs_proj = self.callsign_project(cs_emb)                            # [B, D]
            cs_proj = cs_proj.unsqueeze(1).expand(-1, x.size(1), -1)           # [B, N, D]
            x = x + cs_proj

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
    


    
    