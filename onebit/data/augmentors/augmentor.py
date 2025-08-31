# encoding: utf-8
# author: Yixuan
#
#

from typing import List

import numpy as np

from onebit.config import ConfigManager
from onebit.data.augmentors.factory import AugmentorFactory
from onebit.data.augmentors.base import BaseAugmentor

class Augmentor:
    
    def __init__(self, augmentors: List[BaseAugmentor]):
        self.augmentors = augmentors

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        for augmentor in self.augmentors:
            audio = augmentor(audio)
        return audio
    
    @classmethod
    def from_config(cls, config_manager: ConfigManager) -> 'Augmentor':
        augmentors = AugmentorFactory.create_augmentors(config_manager)
        return Augmentor(augmentors)
