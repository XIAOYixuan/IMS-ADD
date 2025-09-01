# encoding: utf-8
# author: Yixuan
#
#
"""
update the learning rate
support cosine anneal and step lr
"""

from typing import TYPE_CHECKING, Optional
import math

import torch
from torch.optim.lr_scheduler import LRScheduler

from onebit.config import ConfigManager
from onebit.task.callbacks.base import BaseCallback
from onebit.task.callbacks.registry import CallbackRegistry
from onebit.util import get_logger

if TYPE_CHECKING:
    from ..trainer import Trainer

logger = get_logger(__name__)

class CosineAnnealingLR(LRScheduler):
    
    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]

@CallbackRegistry.register('lr_scheduler')
class LRSchedulerCallback(BaseCallback):
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        
        callbacks_config = self.config_manager.get_exp_config().callbacks
        self.scheduler_config = getattr(callbacks_config, 'lr_scheduler', None)
        
        if self.scheduler_config is None:
            self.scheduler_type = 'step'
            self.cosine_T_max = 100  
            self.step_size = 10
            self.step_gamma = 0.5
            self.min_lr_threshold = 1e-6  
        else:
            self.scheduler_type = getattr(self.scheduler_config, 'type', 'cosine')
            # cosine param
            self.cosine_T_max = getattr(self.scheduler_config, 'T_max', 100)
            # step param
            self.step_size = getattr(self.scheduler_config, 'step_size', 30)
            self.step_gamma = getattr(self.scheduler_config, 'gamma', 0.1)
            self.min_lr_threshold = getattr(self.scheduler_config, 'min_lr_threshold', 1e-6)
        
        self.scheduler: Optional[LRScheduler] = None
        self.scheduling_stopped = False  
        
        logger.info(f"LR Scheduler initialized: type={self.scheduler_type}")

    def on_train_begin(self, task: 'Trainer') -> None:
        if self.scheduler_type.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                task.optimizer, 
                T_max=self.cosine_T_max,
                eta_min=self.min_lr_threshold
            )
            logger.info(f"cosine annealing lr scheduler initialized with min_lr={self.min_lr_threshold:.2e}")
        
        elif self.scheduler_type.lower() == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                task.optimizer,
                step_size=self.step_size,
                gamma=self.step_gamma
            )
            logger.info(f"step lr scheduler initialized: step_size={self.step_size}, gamma={self.step_gamma}, min_lr_threshold={self.min_lr_threshold}")
        
        else:
            logger.warning(f"unk scheduler type: {self.scheduler_type}. no lr scheduling then")
            self.scheduler = None

    def on_epoch_end(self, task: 'Trainer') -> None:
        if self.scheduler is not None:
            if self.scheduler_type.lower() == 'step':
                if not self.scheduling_stopped:
                    current_lr = self.scheduler.get_last_lr()[0]  
                    if current_lr <= self.min_lr_threshold:
                        logger.info(f"lr {current_lr} < threshold {self.min_lr_threshold}. Stopping step LR scheduling.")
                        self.scheduling_stopped = True
                        return
                else:
                    return
            
            self.scheduler.step()