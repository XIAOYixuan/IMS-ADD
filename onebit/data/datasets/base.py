# encoding: utf-8
# author: Yixuan
#
#
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
from functools import partial
from transformers import AutoFeatureExtractor

from onebit.config import ConfigManager
from onebit.data.augmentors.augmentor import Augmentor
from onebit.data.postprocessors.postprocessor import PostProcessor
from onebit.util import get_logger

logger = get_logger(__name__)

class BaseDataset(Dataset):

    def __init__(self, 
                split: str,
                config_manager: ConfigManager):
        if config_manager is None:
            raise ValueError("Must provide the config")
            
        self.split = split
        self.config_manager = config_manager
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int):
        pass
    
    def print_stat(self) -> None:
        print(f"Dataset split: {self.split}")
        print(f"Total samples: {len(self)}")