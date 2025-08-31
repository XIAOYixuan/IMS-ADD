# encoding: utf-8
# author: Yixuan
#
#

from typing import List

import numpy as np

from onebit.config import ConfigManager
from onebit.data.postprocessors.factory import PostProcessorFactory
from onebit.data.postprocessors.base import BasePostProcessor

class PostProcessor:
    """
    Main PostProcessor class that orchestrates multiple individual postprocessors.
    """

    def __init__(self, postprocessors: List[BasePostProcessor]):
        self.postprocessors = postprocessors

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        for postprocessor in self.postprocessors:
            audio = postprocessor(audio)
        return audio
    
    @classmethod
    def from_config(cls, config_manager: ConfigManager) -> 'PostProcessor':
        postprocessors = PostProcessorFactory.create_postprocessors(config_manager)
        return PostProcessor(postprocessors)