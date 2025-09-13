# encoding: utf-8
# author: Yixuan
#
#

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import Task

from onebit.config import ConfigManager

class BaseCallback:

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def on_task_begin(self, task: 'Task') -> None:
        pass

    def on_task_end(self, task: 'Task') -> None:
        pass
    
    def on_epoch_begin(self, task: 'Task') -> None:
        pass

    def on_epoch_end(self, task: 'Task') -> None:
        pass

    def on_batch_begin(self, task: 'Task') -> None:
        pass

    def on_batch_end(self, task: 'Task') -> None:
        pass

    def on_train_begin(self, task: 'Task') -> None:
        pass

    def on_train_end(self, task: 'Task') -> None:
        pass
    
    def on_infer_begin(self, task: 'Task') -> None:
        pass

    def on_infer_end(self, task: 'Task') -> None:
        pass

    def on_model_end(self, task: 'Task') -> None:
        pass
