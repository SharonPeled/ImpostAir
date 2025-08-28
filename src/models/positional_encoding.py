import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for patches in the transformer model."""
    
    def __init__(self, d_model: int, max_num_patches: int = 64, encoding_type: str = "learnable"):
        super().__init__()
        self.d_model = d_model
        self.max_num_patches = max_num_patches
        self.encoding_type = encoding_type

        if encoding_type is None:
            return
        elif encoding_type == "learnable":
            self.pos_embedding = nn.Parameter(torch.randn(1, max_num_patches, d_model))
        elif encoding_type == "sinusoidal":
            # Create sinusoidal positional encoding
            pe = torch.zeros(max_num_patches, d_model)
            position = torch.arange(0, max_num_patches, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # Add batch dimension
            
            self.register_buffer('pe', pe)
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, num_patches, d_model]
        
        Returns:
            Tensor of shape [batch_size, num_patches, d_model] with positional encoding added
        """
        if self.encoding_type is None:
            return x
            
        batch_size, num_patches, d_model = x.shape
        
        if num_patches > self.max_num_patches:
            raise ValueError(f"Number of patches ({num_patches}) exceeds maximum ({self.max_num_patches})")
        
        if self.encoding_type == "learnable":
            pos_enc = self.pos_embedding[:, :num_patches, :].clone()
        else:  # sinusoidal
            pos_enc = self.pe[:, :num_patches, :].clone()
        
        return x + pos_enc.expand(batch_size, -1, -1) 