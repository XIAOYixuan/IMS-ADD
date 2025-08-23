# encoding: utf-8
# author: Yixuan
#
#
"""
Define data types for all model classes.

Data flow
FrontendModel -> BackendModel -> Loss
"""

from dataclasses import dataclass

import torch
from transformers.utils.generic import ModelOutput

from onebit.data.audiocollator import AudioBatch

@dataclass
class FrontendOutput:
    foutput: ModelOutput
    attention_mask: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = True):
        # handle the hidden states
        if hasattr(self.foutput, 'last_hidden_state'):
            self.foutput.last_hidden_state = self.foutput.last_hidden_state.to(device, non_blocking=non_blocking) # type: ignore
        if hasattr(self.foutput, 'hidden_states'):
            self.foutput.hidden_states = tuple(
                h.to(device, non_blocking=non_blocking) for h in self.foutput.hidden_states # type: ignore
            )
        if hasattr(self.foutput, 'attention'):
            self.foutput.attentions = tuple(
                a.to(device, non_blocking=non_blocking) for a in self.foutput.attentions # type: ignore
            ) 
        if hasattr(self.foutput, 'extract_features'):
            self.foutput.extract_features = self.foutput.extract_features.to(device, non_blocking=non_blocking) # type: ignore

        self.attention_mask = self.attention_mask.to(device, non_blocking=non_blocking)

        return self


@dataclass
class BackendOutput:
    logits: torch.Tensor
    predictions: torch.Tensor
    frontend_output: FrontendOutput

    def to(self, device: torch.device, non_blocking: bool = True):
        self.logits = self.logits.to(device, non_blocking=non_blocking)
        self.predictions = self.predictions.to(device, non_blocking=non_blocking)
        self.frontend_output.to(device, non_blocking)
        return self
