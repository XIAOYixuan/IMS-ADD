from abc import ABC, abstractmethod

import torch.nn as nn

from onebit.config import ConfigManager
from onebit.util import get_logger
from onebit.data import AudioBatch
logger = get_logger(__name__)

class BaseHook(ABC):

    def __init__(self, config_manager: ConfigManager, model: nn.Module):
        self.config_manager = config_manager
        self.model = model

    def on_batch_end(self, *args, **kwargs) -> None:
        pass