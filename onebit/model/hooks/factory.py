# encoding: utf-8
# author: Yixuan
#
#
import torch.nn as nn

from onebit.config import ConfigManager
from onebit.model.hooks.base import BaseHook
from onebit.model.hooks.registry import HookRegistry

from onebit.util import get_logger
logger = get_logger(__name__)

class HookFactory:

    @staticmethod
    def create_hooks(config_manager: ConfigManager, model: nn.Module):
        hook_config = getattr(config_manager.get_model_config(), 'hooks', None)
        logger.info(f"hook_config: {hook_config}")
        hooks = []
        if hook_config is None:
            return hooks

        for hook_name in hook_config.keys():
            hook = HookFactory.create(hook_name, config_manager, model)
            hooks.append(hook)

        return hooks

    @staticmethod
    def create(hook_name: str, config_manager: ConfigManager, model: nn.Module) -> BaseHook:
        HookRegistry._ensure_initialized()
        
        if HookRegistry.has_hook(hook_name):
            hook_cls = HookRegistry.get(hook_name)
            return hook_cls(config_manager, model)
        else:
            raise ValueError(f"Unknown hook: {hook_name}. Available hooks: {HookRegistry.list_hooks()}")

