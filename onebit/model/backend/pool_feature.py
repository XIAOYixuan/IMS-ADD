# encoding: utf-8
# author: Yixuan
#
#
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from onebit.config import ConfigManager
from onebit.data import AudioBatch
from onebit.model.backend.base import BaseBackendModel
from onebit.model.datatypes import FrontendOutput, BackendOutput
from onebit.model.backend.registry import BackendRegistry
from onebit.model.operators import MeanPoolAllLayers

@dataclass
class PoolFeatureOutput(BackendOutput):
    pooled_feat: torch.Tensor


@BackendRegistry.register("pool_feature")
class PoolFeature(BaseBackendModel):
    """
    Simply pass through features with mean pooling 
    Output would be [bz, dim]
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        model_config = config_manager.get_model_config().backend
        self.pool_type = getattr(model_config, 'pool_type', 'mean')
        if self.pool_type == 'mean':
            self.pool_layer = MeanPoolAllLayers()

    def forward(self, audio_batch: AudioBatch, frontend_output: FrontendOutput) -> PoolFeatureOutput:
        """
        Layer-wise
        Output: [B, ]
        """
        # Tuple[torch.Tensor] size: numble of layers, tensor shape : [bz, T, D]
        hidden_states: Tuple[torch.Tensor] = frontend_output.foutput.hidden_states # type: ignore
        # shape: [bz, T]
        attention_mask = frontend_output.attention_mask
        # L, B, T, D
        hs_stack = torch.stack(hidden_states, dim=0)
        L, B, T, D = hs_stack.shape
        pooled = self.pool_layer(hs_stack, attention_mask) 
        logits = pooled.mean(dim=0).mean(dim=1) # [B, ]

        return PoolFeatureOutput(
            logits=logits,
            predictions=logits,
            frontend_output=frontend_output,
            pooled_feat=pooled
        )


if __name__ == '__main__':
    B, T = 3, 64000
    input_values = torch.randn((B, T), dtype=torch.float32)
    attention_mask = torch.ones((B, T), dtype=torch.bool)
    label_tensors = torch.zeros((B,), dtype=torch.long)

    t, d = 199, 768
    from transformers.utils.generic import ModelOutput
    hf_model_out = ModelOutput(
        last_hidden_state=torch.randn(B, t, d),
        hidden_states=tuple(torch.randn(B, t, d) for _ in range(13)) 
    )

    audio_batch = AudioBatch(
        input_values=input_values,
        attention_mask=attention_mask,
        label_tensors=label_tensors,
        labels = [''],
        uttids=[''],
        speakers=[''],
        attackers=[''],
        origin_ds=[''],
        audio_paths=[''], 
    )
    frontend_out = FrontendOutput(
        foutput=hf_model_out,
        attention_mask=torch.ones((B, t))
    )

    device = torch.device('cuda')
    #device = torch.device('cpu') 
    config_path = "onebit/configs/test.yaml"
    config_manager = ConfigManager(config_path)
    model = PoolFeature(config_manager)

    model.to(device)
    audio_batch = audio_batch.to(device)

    model.eval()
    with torch.no_grad():
        model_out: PoolFeatureOutput = model(audio_batch, frontend_out)
    
    print("logits shape", model_out.logits.shape)
    print("pooled feat", model_out.pooled_feat.shape)
