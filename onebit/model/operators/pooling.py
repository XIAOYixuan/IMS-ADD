# encoding: utf-8
# author: Yixuan
#
#

from typing import Optional

import torch
import torch.nn as nn

class MeanPoolAllLayers(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, 
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        # hs_stack: [L, B, T, D]
        # attention_mask: [B, T]
        
        if attention_mask is not None:
            mask = attention_mask.float() # [B, T]
            # sum over T dimension, [L, B, D]
            summed = torch.einsum('l b t d, b t -> l b d', hidden_states, mask)
            # counst for each layer: [B] -> [1, B, 1]
            counts = mask.sum(dim=1) .clamp(min=1)[None, :, None]
            pooled = summed / counts # [L, B, D]
        else:
            # hs_stack [L, B, T, D] -> [L, B, D]
            pooled = hidden_states.mean(dim=2) 
        
        return pooled
    