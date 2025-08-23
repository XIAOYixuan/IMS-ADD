# encoding: utf-8
# author: Yixuan
#
#
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from onebit.config import ConfigManager
from onebit.data.audiocollator import AudioBatch
from onebit.model import frontend
from onebit.model.backend.base import BaseBackendModel
from onebit.model.datatypes import FrontendOutput, BackendOutput
from onebit.model.backend.registry import BackendRegistry
from onebit.model.operators import MeanPoolAllLayers

@dataclass
class DirectProjectorOutput(BackendOutput):
    pass

@BackendRegistry.register("direct_projector")
class DirectProjector(BaseBackendModel):
    """
    Directly project the last hidden states to num of classes
    1FFN + 1Proj
    D -> 256 -> 1
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        model_config = config_manager.get_model_config().backend
        
        # Get input dimension from config or use default
        self.input_dim = getattr(model_config, 'input_dim', 768)
        self.hidden_dim = getattr(model_config, 'hidden_dim', 256)
        self.num_classes = getattr(model_config, 'num_classes', 1)
        
        self.ffn = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = nn.ReLU()
        self.projector = nn.Linear(self.hidden_dim, self.num_classes)
        
    def forward(self, audio_batch: AudioBatch, frontend_output: FrontendOutput) -> DirectProjectorOutput:
        last_hidden_state = frontend_output.foutput.last_hidden_state  # type: ignore
        # attn mask [B, T]
        attention_mask = frontend_output.attention_mask
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()

        # mean pool 
        masked_hidden_state = last_hidden_state * attention_mask_expanded
        pooled_output = masked_hidden_state.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()
        
        hidden = self.ffn(pooled_output)
        hidden = self.activation(hidden)
        logits = self.projector(hidden)
        
        if self.num_classes == 1:
            logits = logits.squeeze(-1) # turn [B, n_cls] -> [B,]
        
        predictions = logits
        
        return DirectProjectorOutput(
            logits=logits,
            predictions=predictions,
            frontend_output=frontend_output
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
    model = DirectProjector(config_manager)

    model.to(device)
    audio_batch = audio_batch.to(device)
    frontend_out.to(device)

    model.eval()
    with torch.no_grad():
        model_out: DirectProjectorOutput = model(audio_batch, frontend_out)
    
    print("logits shape", model_out.logits.shape)
    print("predictions shape", model_out.predictions.shape)
