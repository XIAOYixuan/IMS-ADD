# encoding: utf-8
# author: Yixuan
#
#

from onebit.config import ConfigManager
from onebit.data.collators.base import BaseCollator
from onebit.data.collators.registry import CollatorRegistry

class CollatorFactory:

    @staticmethod
    def create(config_manager: ConfigManager) -> BaseCollator:
        collator_cfg = config_manager.get_data_config().get('collator', {})
        collator_name: str = getattr(collator_cfg, 'name', 'audio')
        
        CollatorRegistry._ensure_initialized()
        
        if CollatorRegistry.has_collator(collator_name):
            collator_cls = CollatorRegistry.get(collator_name)
            return collator_cls(config_manager)
        else:
            raise ValueError(f"Unknown collator: {collator_name}. Available collators: {CollatorRegistry.list_collators()}")