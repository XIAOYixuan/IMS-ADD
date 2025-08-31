# encoding: utf-8
# author: Yixuan
#
#

import numpy as np

from onebit.config import ConfigManager
from onebit.data.postprocessors.base import BasePostProcessor
from onebit.data.postprocessors.registry import PostProcessorRegistry

@PostProcessorRegistry.register('fixed_length')
class FixedLengthPostProcessor(BasePostProcessor):
    """
    Fix audio has a fixed length by 
    1. repeating short audio or 
    2. randomly cropping long audio.
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)

    def _random_st_for_long_audio(self, audio: np.ndarray) -> np.ndarray:
        # audio shape: [length]
        if audio.shape[0] <= self.max_samples:
            return audio
        
        stt = np.random.randint(audio.shape[0] - self.max_samples)
        audio = audio[stt: stt+self.max_samples]
        return audio

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        # audio shape: [length]
        alen = audio.shape[0]
        if alen == self.max_samples:
            return audio
        elif alen < self.max_samples:
            num_repeats = -(-self.max_samples//alen) 
            audio = np.tile(audio, num_repeats) 
        audio = self._random_st_for_long_audio(audio)
        return audio
