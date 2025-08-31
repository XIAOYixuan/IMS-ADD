# encoding: utf-8
# author: Yixuan
#
#

import numpy as np

from onebit.config import ConfigManager
from onebit.data.postprocessors.base import BasePostProcessor
from onebit.data.postprocessors.registry import PostProcessorRegistry

@PostProcessorRegistry.register('random_st_for_long_audio')
class RandomStartPostProcessor(BasePostProcessor):
    """
    Randomly selects a starting position for long audio 
    to fit within max_samples.
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        # audio shape: [length]
        if audio.shape[0] <= self.max_samples:
            return audio
        
        stt = np.random.randint(audio.shape[0] - self.max_samples)
        audio = audio[stt: stt+self.max_samples]
        return audio
