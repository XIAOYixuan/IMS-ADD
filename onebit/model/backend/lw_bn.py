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
from onebit.data.audiocollator import AudioBatch
from onebit.model import frontend
from onebit.model.backend.base import BaseBackendModel
from onebit.model.datatypes import FrontendOutput, BackendOutput
from onebit.model.backend.registry import BackendRegistry
from onebit.model.operators import MeanPoolAllLayers

@dataclass
class LWBNOutput(BackendOutput):
    """
    We don't use logits, we only provide prediction which is the 
    cos distance to the bonafide centers
    """
    bonafide_centers: torch.Tensor
    scores: torch.Tensor


@BackendRegistry.register("lw_bn")
class LWBN(BaseBackendModel):
    """
    Re-implementation of LWBN model mentioned in my 
    Layer-wise Decision Fusion .... paper (interspeech25)
    """

    def __init__(self, config_manager):
        super().__init__(config_manager)
        model_config = config_manager.get_model_config().backend

        self.input_dim = getattr(model_config, "input_dim", 1024) # ssl dim
        self.layer_attn_head = getattr(model_config, "layer_attn_head", 8) 
        self.num_layers = getattr(model_config, "num_layers", 25)
        # L, D, A parameters
        self.time_attn_mlp1 = nn.Parameter(torch.Tensor(self.num_layers, self.input_dim, self.layer_attn_head))
        self.bottleneck_dim = getattr(model_config, "bottleneck_dim", 256)

        self.bottleneck_proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.input_dim, self.bottleneck_dim)
        )
        nn.init.xavier_uniform_(self.time_attn_mlp1)
        
        # init vector for oc softmax
        init_vector = torch.empty(self.num_layers, self.bottleneck_dim)
        nn.init.kaiming_uniform_(init_vector, a=0.25)
        self.center = nn.Parameter(F.normalize(init_vector, p=2, dim=1))
        self.softplus = nn.Softplus()        

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
        # L, N, d
        bn_feat = self.bottleneck_proj(utt_h)
        # batch_size, l, feat_dim
        # N, L, D
        bn_feat = rearrange(bn_feat, 'l n d -> n l d')

        w = F.normalize(self.center, p=2, dim=1) # [L, D]
        x = F.normalize(bn_feat, p=2, dim=2) # [N, L, D]

        # N, L, D -> N, L
        scores = (x * w.unsqueeze(0)).sum(dim=-1)
        predictions = scores.sum(dim=-1)
        return LWBNOutput(
            logits=scores,
            predictions=predictions,
            frontend_output=frontend_output,
            bonafide_centers=self.center,
            scores=scores
        )
