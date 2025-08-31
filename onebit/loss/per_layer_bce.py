# encoindg: utf-8
# author: Yixuan
#
#
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from onebit.loss.base import BaseLoss
from onebit.loss.datatypes import LossOutput 
from onebit.loss.registry import LossRegistry
from onebit.data import AudioBatch
from onebit.model.datatypes import FrontendOutput, BackendOutput
from onebit.util import get_logger

logger = get_logger(__name__)

@dataclass
class PerLayerBCELossOutput(LossOutput):
    layer_losses: List[float]

@LossRegistry.register('per_layer_bce')
class PerLayerBCELoss(BaseLoss):

    """
    This class accepts an input logit with shape [L, B].
        L for num of layers, B for batch size
    Then it applies BCEWithLogitsLoss to each layer
    """
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.num_layers = config_manager.get_model_config().backend.num_layers
        loss_config = self.config_manager.get_loss_config()
        real_weight = loss_config.get('real_weight', 1.0)
        logger.info(f"Using real_weight = {real_weight}")
        real_weight = torch.Tensor([real_weight])
        self.loss_fns = nn.ModuleList(
            [nn.BCEWithLogitsLoss(pos_weight=real_weight) 
             for _ in range(self.num_layers)])

    def forward(self, audio_batch: AudioBatch, back_out: BackendOutput):
        logits = back_out.logits # [L, B]
        L, B = logits.shape
        if L != self.num_layers:
            raise ValueError(f'dim mismatch, L {L} vs nun layers {self.num_layers}')
        labels = audio_batch.label_tensors # [B]

        total_loss = logits.new_zeros(())
        layer_losses = []
        for i in range(len(self.loss_fns)):
            loss_i = self.loss_fns[i](logits[i, :], labels.float())
            total_loss += loss_i
            layer_losses.append(loss_i.detach().item())

        total_loss = total_loss / self.num_layers
        return PerLayerBCELossOutput(
            loss=total_loss,
            layer_losses=layer_losses
        )