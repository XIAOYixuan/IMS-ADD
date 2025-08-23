# encoding: utf-8
# author: Yixuan
#
#

from typing import List

import numpy as np

from onebit.config import ConfigManager

class PostProcessor:

    def __init__(self, 
                 process_types: List[str], 
                 config_manager: ConfigManager):
        self.process_types = process_types
        self.config_manager = config_manager
        data_conf = self.config_manager.get_data_config()
        sample_rate = data_conf.dataset.sample_rate # type: ignore 
        max_length = data_conf.dataset.max_length
        self.max_samples = int(sample_rate * max_length)

    def _random_st_for_long_audio(self, audio: np.ndarray) -> np.ndarray:
        # audio shape: [length]
        if audio.shape[0] <= self.max_samples:
            return audio
        
        stt = np.random.randint(audio.shape[0] - self.max_samples)
        audio = audio[stt: stt+self.max_samples]
        return audio

    def _fixed_length(self, audio: np.ndarray) -> np.ndarray:
        # audio shape: [length]
        alen = audio.shape[0]
        if alen == self.max_samples:
            return audio
        elif alen < self.max_samples:
            num_repeats = int(self.max_samples / alen) + 1
            audio = audio.repeat(0, num_repeats)
        audio = self._random_st_for_long_audio(audio)
        return audio

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        for process_type in self.process_types:
            if process_type == 'random_st_for_long_audio':
                audio = self._random_st_for_long_audio(audio)
            elif process_type == 'fixed_length':
                audio = self._fixed_length(audio)
        return audio
    
    @classmethod
    def from_config(cls, config_manager: ConfigManager) -> 'PostProcessor':
        post_config = config_manager.get_post_processing_config()
        return PostProcessor(
            post_config.types,
            config_manager
        )