# encoding: utf-8
# author: Yixuan
#
#

from abc import ABC, abstractmethod

import torch.nn as nn

from onebit.config import ConfigManager
from onebit.data import AudioBatch
from onebit.model.datatypes import FrontendOutput, BackendOutput
from onebit.loss.datatypes import LossOutput


class BaseLoss(nn.Module, ABC):

    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager = config_manager

    @abstractmethod
    def forward(self, 
                audio_batch: AudioBatch, 
                back_out: BackendOutput) -> LossOutput:
        pass