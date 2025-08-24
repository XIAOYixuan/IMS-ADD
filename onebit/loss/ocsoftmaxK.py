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
from onebit.data.audiocollator import AudioBatch
from onebit.model.datatypes import FrontendOutput, BackendOutput
from onebit.model.backend.lw_bn import FrontendOutput, LWBNOutput 
from onebit.util import get_logger
from onebit.config import ConfigManager

logger = get_logger(__name__)

@LossRegistry.register('ocsoftmaxk')
class OCSoftmaxK(BaseLoss):
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        loss_config = self.config_manager.get_loss_config()
        self.m_real = getattr(loss_config, 'm_real', 0.9)
        self.m_fake = getattr(loss_config, 'm_fake', 0.2)
        self.alpha = getattr(loss_config, 'alpha', 20.0)
        self.real_weight = loss_config.get('real_weight', 1.0)
        self.softplus = nn.Softplus()
        # fake weight default 1

    def forward(self, audio_batch: AudioBatch, back_out: BackendOutput):
        labels = audio_batch.label_tensors # batch size
        # bonafide: 1
        # spoof: 0
        fake_mask = labels == 0
        real_mask = labels == 1

        scores = back_out.logits
        # [num_negative, k] and [num_positive, k]
        fake_scores = scores[fake_mask]
        real_scores = scores[real_mask]

        total_loss = torch.tensor(0.0, device=real_scores.device, dtype=real_scores.dtype)
        if real_scores.numel() > 0:
            real_loss = self.real_weight * self.softplus(self.alpha * (self.m_real - real_scores)).mean(dim=0)
            total_loss += real_loss.sum()
        if fake_scores.numel() > 0:
            fake_loss = self.softplus(self.alpha * (fake_scores - self.m_fake)).mean(dim=0)
            total_loss += fake_loss.sum()
        
        return LossOutput(loss=total_loss)
