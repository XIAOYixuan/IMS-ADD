# encoding: utf-8
# author: Yixuan
#
#

from typing import List

import torch.nn as nn

from onebit.config import ConfigManager
from onebit.model.hooks.base import BaseHook
from onebit.model.hooks.factory import HookFactory

class Hook:

    def __init__(self, hooks: List[BaseHook]):
        self.hooks = hooks

    @classmethod
    def from_config(cls, 
                    config_manager: ConfigManager, 
                    model: nn.Module) -> 'Hook':
        hooks = HookFactory.create_hooks(config_manager, model)
        return Hook(hooks)
    
    def on_batch_end(self, *args, **kwargs) -> None:
        for hook in self.hooks:
            hook.on_batch_end(*args, **kwargs)