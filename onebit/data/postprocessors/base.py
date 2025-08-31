from abc import ABC, abstractmethod

import numpy as np

from onebit.config import ConfigManager
from onebit.util import get_logger
logger = get_logger(__name__)

class BasePostProcessor(ABC):

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        data_conf = self.config_manager.get_data_config()
        sample_rate = data_conf.dataset.sample_rate # type: ignore 
        max_length = data_conf.dataset.max_length
        max_samples = int(sample_rate * max_length)
        self.max_samples = getattr(data_conf.dataset, 'max_sample', max_samples)
        logger.info(f"Max sample: {self.max_samples}")

    @abstractmethod
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        pass
