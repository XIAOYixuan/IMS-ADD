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
from onebit.model.backend.layer_wise import LayerWiseModel, LayerWiseOutput
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
class LWBN(LayerWiseModel):
    """
    Re-implementation of LWBN model mentioned in my 
    Layer-wise Decision Fusion .... paper (interspeech25)
    """

    def __init__(self, config_manager):
        super().__init__(config_manager)
        model_config = config_manager.get_model_config().backend

        self.bottleneck_dim = getattr(model_config, "bottleneck_dim", 256)

        self.bottleneck_proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.input_dim, self.bottleneck_dim)  # type: ignore
        )
        
        # init vector for oc softmax
        init_vector = torch.empty(self.num_layers, self.bottleneck_dim)  # type: ignore
        nn.init.kaiming_uniform_(init_vector, a=0.25)
        self.center = nn.Parameter(F.normalize(init_vector, p=2, dim=1))
        self.softplus = nn.Softplus()        

    def forward(self, audio_batch, frontend_output):
        # Get base layer-wise features from parent class
        base_output: LayerWiseOutput = super().forward(audio_batch, frontend_output)  # type: ignore
        utt_h = base_output.utt_features  # L, N, D
        
        bn_feat = self.bottleneck_proj(utt_h)
        # N, L, D
        bn_feat = rearrange(bn_feat, 'l n d -> n l d')

        w = F.normalize(self.center, p=2, dim=1) # [L, D]
        x = F.normalize(bn_feat, p=2, dim=2) # [N, L, D]

        # N, L, D -> N, L
        scores = (x * w.unsqueeze(0)).sum(dim=-1)
        predictions = scores.clone().sum(dim=-1)
        return LWBNOutput(
            logits=x,
            predictions=predictions,
            frontend_output=frontend_output,
            bonafide_centers=self.center,
            scores=scores
        )
