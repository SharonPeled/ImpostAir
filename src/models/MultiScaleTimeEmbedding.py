import torch
import torch.nn as nn

from src.models.Time2Vec import Time2Vec


class MultiScaleTimeEmbedding(nn.Module):
    """
    Multi-scale Time2Vec over periodic calendar phases, concatenated to size d_model.

    Args:
      d_model: final concatenated embedding size
      ref_timestamp: reference timestamp for normalization, in seconds
      scales: which periodic scales to include (see above)
    """
    def __init__(self, d_model: int, ref_timestamp: int, scales: list[str] = ['hours', 'days', 'months', 'years']):
        super().__init__()
        self.d_model = d_model
        k_list = chunk_quantity(d_model, len(scales))
        self.time2vec_modules = nn.ModuleList([Time2Vec(k=k, 
                                                        ref_timestamp=ref_timestamp, 
                                                        scale=scale) for scale, k in zip(scales, k_list)])

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return torch.cat([t2v(tau) for t2v in self.time2vec_modules], dim=-1)


def chunk_quantity(x, y):
    base = x // y
    remainder = x % y
    return [base + 1 if i < remainder else base for i in range(y)]