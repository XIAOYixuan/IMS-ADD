# encoding: utf-8
# author: Yixuan
#
#
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from onebit.config import ConfigManager
from onebit.data import AudioBatch
from onebit.model import frontend
from onebit.model.backend.base import BaseBackendModel
from onebit.model.datatypes import FrontendOutput, BackendOutput
from onebit.model.backend.registry import BackendRegistry
from onebit.model.operators import MeanPoolAllLayers

@dataclass
class LayerWiseOutput(BackendOutput):
    utt_features: torch.Tensor  # L, N,D 


@BackendRegistry.register("layer_wise")
class LayerWiseModel(BaseBackendModel):

    def __init__(self, config_manager):
        super().__init__(config_manager)
        model_config = config_manager.get_model_config().backend

        self.input_dim = getattr(model_config, "input_dim", 1024) # ssl dim
        self.layer_attn_head = getattr(model_config, "layer_attn_head", 8) 
        self.num_layers = getattr(model_config, "num_layers", 25)
        # L, D, A parameters
        self.time_attn_mlp1 = nn.Parameter(torch.Tensor(self.num_layers, self.input_dim, self.layer_attn_head))
        nn.init.xavier_uniform_(self.time_attn_mlp1)        

    def forward(self, audio_batch, frontend_output):
        hidden_states: Tuple[torch.Tensor] = frontend_output.foutput.hidden_states # type: ignore
        # L, N, T, D
        hs_stack = torch.stack(hidden_states, dim=0)
        L, N, T, D = hs_stack.shape
        hs_stack = rearrange(hs_stack, 'l n t d -> l (n t) d')

        # L, X, D * L, D, A -> L, X(N*T), A
        layer_attn = torch.bmm(hs_stack, self.time_attn_mlp1)
        # L, X(N*T)
        time_score = torch.logsumexp(layer_attn, dim=-1) 
        # L, N, T
        time_score = rearrange(time_score, 'l (n t) -> l n t', n=N, t=T)
        # L, N, T
        time_score = F.softmax(time_score, dim=-1)

        hs_stack = hs_stack.view(L, N, T, D)
        # L, N, T, D
        weighted_h = hs_stack * time_score.unsqueeze(-1)
        # L, N, D
        utt_h = torch.sum(weighted_h, dim=2)
        
        return LayerWiseOutput(
            logits=utt_h, # placeholder
            predictions=utt_h, # placeholder
            frontend_output=frontend_output,
            utt_features=utt_h
        )
