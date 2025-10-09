# encoding: utf-8
# author: Yixuan
#
#
import torch.nn as nn
from abc import ABC, abstractmethod

from onebit.config import ConfigManager
from onebit.data import AudioBatch
from onebit.model.datatypes import FrontendOutput

class BaseFrontendModel(nn.Module, ABC):

    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager = config_manager

    @abstractmethod
    def forward(self, audio_batch: AudioBatch) -> FrontendOutput:
        pass