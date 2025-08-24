# encoding: utf-8
# author: Yixuan
#
#
from typing import List

from onebit.task.callbacks.base import BaseCallback
from onebit.task.callbacks.factory import CallbackFactory
from onebit.config import ConfigManager

class Task:

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.callback_cfgs = self.config_manager.get_exp_config().callbacks
        self.user_callback_names: List[str] = []
        self.builtin_callback_names: List[str] = []
    
    def _create_callbacks(self) -> List[BaseCallback]:
        created_callback_names = set()
        set_random_seed = 'set_random_seed'
        callbacks = [CallbackFactory.create('set_random_seed', self.config_manager)]
        created_callback_names.add(set_random_seed)

        # user callbacks has higher priority
        # and the callbacks will be executed in the same order they appear in the cfg
        if self.callback_cfgs is not None:
            for cb_name in self.callback_cfgs.keys():
                if self.callback_cfgs[cb_name] is None: continue
                callback = CallbackFactory.create(cb_name, self.config_manager)
                callbacks.append(callback)
                created_callback_names.add(cb_name)
        
        for cb_name in self.builtin_callback_names:
            if cb_name not in created_callback_names:
                callbacks.append(CallbackFactory.create(cb_name, self.config_manager))
                created_callback_names.add(cb_name)
        
        return callbacks

    def start(self):
        pass