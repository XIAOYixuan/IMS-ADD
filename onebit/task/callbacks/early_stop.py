# encoding: utf-8
# author: Yixuan
#
#
"""
early stopping, min_delta and patience
"""

from typing import TYPE_CHECKING

from onebit.config import ConfigManager
from onebit.task.callbacks.base import BaseCallback
from onebit.task.callbacks.registry import CallbackRegistry
from onebit.util import get_logger

if TYPE_CHECKING:
    from ..trainer import Trainer

logger = get_logger(__name__)

@CallbackRegistry.register('early_stop')
class EarlyStopCallback(BaseCallback):
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        
        callbacks_config = self.config_manager.get_exp_config().callbacks
        early_stop_config = getattr(callbacks_config, 'early_stop', None)
        
        if early_stop_config is None:
            self.patience = 10
            self.min_delta = 1e-3
        else:
            self.patience = getattr(early_stop_config, 'patience', 10)
            self.min_delta = getattr(early_stop_config, 'min_delta', 1e-3)
        
        self.epochs_without_improvement = 0
        self.stopped_epoch = 0
        self.best_eer = float('inf')
        
        logger.info(f"early stopping initialized: patience={self.patience}, min_delta={self.min_delta}")

    def on_epoch_end(self, task: 'Trainer') -> None:
        current_eer = task.cur_val_status.eer
        
        if current_eer < self.best_eer - self.min_delta:
            self.best_eer = current_eer
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        if self.epochs_without_improvement >= self.patience:
            task.should_stop = True