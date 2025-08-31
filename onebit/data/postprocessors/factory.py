# encoding: utf-8
# author: Yixuan
#
#

from typing import List
from onebit.config import ConfigManager
from onebit.data.postprocessors.registry import PostProcessorRegistry
from onebit.data.postprocessors.base import BasePostProcessor

class PostProcessorFactory:

    @staticmethod
    def create_postprocessors(config_manager: ConfigManager):
        post_config = config_manager.get_post_processing_config()

        postprocessors = []
        if post_config is None:
            return postprocessors

        for postprocessor_name in post_config.keys():
            postprocessor = PostProcessorFactory.create(str(postprocessor_name), config_manager)
            postprocessors.append(postprocessor)
        
        return postprocessors

    @staticmethod
    def create(postprocessor_name: str, config_manager: ConfigManager) -> BasePostProcessor:
        PostProcessorRegistry._ensure_initialized()
        
        if PostProcessorRegistry.has_postprocessor(postprocessor_name):
            postprocessor_cls = PostProcessorRegistry.get(postprocessor_name)
            return postprocessor_cls(config_manager)
        else:
            raise ValueError(f"Unknown postprocessor: {postprocessor_name}. Available postprocessors: {PostProcessorRegistry.list_postprocessors()}")