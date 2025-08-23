# encoding: utf-8
# author: Yixuan
#
#

from typing import List, Dict, Any

from onebit.config import ConfigManager
from onebit.task.callbacks.base import BaseCallback
from onebit.task.callbacks.registry import CallbackRegistry

class CallbackFactory:

    @staticmethod
    def create(callback_name: str, config_manager: ConfigManager) -> BaseCallback:
        CallbackRegistry._ensure_initialized()
        
        if CallbackRegistry.has_callback(callback_name):
            callback_class = CallbackRegistry.get(callback_name)
            return callback_class(config_manager)
        else:
            raise ValueError(f"Unknown callback: {callback_name}. Available callbacks: {CallbackRegistry.list_callbacks()}")