import torch
import torch.nn as nn


class CallsignEmbedding(nn.Module):
    """Embedding for callsign."""
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, callsign_ids: torch.Tensor) -> torch.Tensor:
        # callsign_ids: [batch_size]
        # Returns: [batch_size, d_model]
        return self.embedding(callsign_ids)