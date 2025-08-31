# encoding: utf-8
# author: Yixuan
#
#
import torch.nn as nn
from abc import ABC, abstractmethod

from onebit.config import ConfigManager
from onebit.model.datatypes import FrontendOutput, BackendOutput
from onebit.data import AudioBatch

class BaseBackendModel(nn.Module, ABC):

    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager = config_manager

    @abstractmethod
    def forward(self, audio_batch: AudioBatch, frontend_output: FrontendOutput) -> BackendOutput:
        pass