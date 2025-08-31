# encoding: utf-8
# author: Yixuan
#
#

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from functools import partial
from transformers import AutoFeatureExtractor

from onebit.config import ConfigManager

@dataclass
class BaseAudioBatch:
    
    def to(self, device: torch.device, non_blocking: bool = True):
        for fld, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, fld, value.to(device, non_blocking=non_blocking))
        return self

    def asdict(self) -> Dict[str, Any]:
        return self.__dict__

class BaseCollator(ABC):

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    @abstractmethod
    def __call__(self, batch: List[Any]) -> BaseAudioBatch:
        pass