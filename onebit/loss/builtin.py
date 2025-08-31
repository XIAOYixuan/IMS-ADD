# encoindg: utf-8
# author: Yixuan
#
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from onebit.loss.base import BaseLoss
from onebit.loss.datatypes import LossOutput 
from onebit.loss.registry import LossRegistry
from onebit.data import AudioBatch
from onebit.model.datatypes import FrontendOutput, BackendOutput

@LossRegistry.register('bcewithlogits')
class BCEWithLogitsLoss(BaseLoss):

    def __init__(self, config_manager):
        super().__init__(config_manager)
        loss_config = self.config_manager.get_loss_config()
        real_weight = loss_config.get('real_weight', 1.0)
        self.real_weight = torch.Tensor([real_weight])

        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.real_weight)

    def forward(self, audio_batch: AudioBatch, 
                    back_out: BackendOutput) -> LossOutput:
        logits = back_out.logits # shape
        labels = audio_batch.label_tensors # shape
        
        if torch.isinf(logits).any() or torch.isnan(logits).any():
            raise ValueError(f"NaN exists in Logits")
        
        loss = self.loss(logits, labels.float())
        return LossOutput(
            loss=loss
        )
