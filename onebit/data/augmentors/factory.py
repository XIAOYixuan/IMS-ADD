# encoding: utf-8
# author: Yixuan
#
#
from typing import List, Union
from onebit.config import ConfigManager
from onebit.data.augmentors.registry import AugmentorRegistry 
from onebit.data.augmentors.base import BaseAugmentor

class AugmentorFactory:

    # TODO: 
    @staticmethod
    def create_augmentors(config_manager: ConfigManager):
        augmentor_cfgs = config_manager.get_data_config().aug

        augmentors = []
        if augmentor_cfgs is None:
            return augmentors

        for augmentor_name in augmentor_cfgs.keys():
            augmentor = AugmentorFactory.create(augmentor_name, config_manager)
            augmentors.append(augmentor)
        
        return augmentors

    @staticmethod
    def create(augmentor_name: str, config_manager: ConfigManager) -> BaseAugmentor:
        AugmentorRegistry._ensure_initialized()
        
        if AugmentorRegistry.has_augmentor(augmentor_name):
            augmentor_cls = AugmentorRegistry.get(augmentor_name)
            return augmentor_cls(config_manager)
        else:
            raise ValueError(f"Unknown backend: {augmentor_name}. Available backends: {AugmentorRegistry.list_augmentors()}")