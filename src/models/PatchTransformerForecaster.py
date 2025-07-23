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
        super(PatchTransformerForecaster, self).__init__(config)
        
        # Model architecture parameters
        self.d_model = config['model']['d_model']
        self.n_heads = config['model']['n_heads']
        self.n_layers = config['model']['n_layers']
        self.d_ff = config['model']['d_ff']
        self.dropout = config['model']['dropout']
        self.activation = config['model']['activation']
        self.num_features = config['model']['num_features']
        self.max_num_patches = config['model']['max_num_patches']
        self.context_length = config['model']['context_length']
        self.pos_encoding_type = config['model']['pos_encoding_type']
        
        # Build model
        self._build_model()
        
        # Initialize weights
        self._init_weights()
        
        print(f"✓ Model initialized PatchTransformerForecaster.")
    
    def _build_model(self):
        """Build the patch-based transformer architecture."""
        
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            num_features=self.num_features,
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
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=self.n_layers
        )
        
        # Next patch prediction head
        patch_output_dim = self.patch_len * self.num_features
        self.next_patch_head = nn.Linear(self.d_model, patch_output_dim)
        
        # Layer normalization for final output
        self.output_norm = nn.LayerNorm(self.d_model)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the patch transformer.
        
        Args:
            context: [batch_size, context_steps, num_features]
        
        Returns:
            next_patch_pred: [batch_size, patch_len * num_features]
        """
        context_patches = divide_ts_into_patches(context, self.patch_len)
        batch_size, num_patches, patch_dim = context_patches.shape
        
        # Patch embedding + positional encoding
        embedded = self.patch_embedding(context_patches)  # [batch, num_patches, d_model]
        embedded = self.pos_encoding(embedded)  # [batch, num_patches, d_model]
        
        causal_mask = torch.tril(torch.ones(
            num_patches, num_patches, 
            device=context_patches.device
        ))
        
        # Pass through transformer decoder
        # In decoder-only mode
        transformer_output = self.transformer_decoder(
            tgt=embedded,
            memory=embedded,
            tgt_mask=causal_mask,
            memory_mask=causal_mask
        )  # [batch, num_patches, d_model]
        
        # Extract final patch representation (last position)
        final_repr = transformer_output[:, -1, :]  # [batch, d_model]
        final_repr = self.output_norm(final_repr)
        
        # Predict next patch
        next_patch_pred = self.next_patch_head(final_repr)  # [batch, patch_len * num_features]
        
        return next_patch_pred.reshape(batch_size, self.patch_len, self.num_features)


    
    
    