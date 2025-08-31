from abc import ABC, abstractmethod

import numpy as  np

from onebit.config import ConfigManager
class BaseAugmentor(ABC):

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    @abstractmethod
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        pass
    