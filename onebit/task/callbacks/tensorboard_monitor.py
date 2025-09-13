# encoding: utf-8
# author: Yixuan
#
#

"""
log the status every tensorboard_log_interval (default 40 batches)
log the loss and eer
"""

from typing import TYPE_CHECKING, Optional 

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from onebit.config import ConfigManager
from onebit.task.callbacks.base import BaseCallback
from onebit.task.callbacks.registry import CallbackRegistry
from onebit.util import get_logger

if TYPE_CHECKING:
    from ..trainer import Trainer

logger = get_logger(__name__)

@CallbackRegistry.register('tensorboard_monitor')
class TensorboardMonitorCallback(BaseCallback):

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        
        exp_config = self.config_manager.get_exp_config()
        self.log_interval = getattr(exp_config, 'tensorboard_log_interval', 40)
        
        root_path = getattr(exp_config, 'root_dir', None) 
        if root_path is None:
            root_path = str(Path(__file__).parent.parent.parent.parent)
        self.root_dir = Path(root_path)
        self.log_dir = self.root_dir / "tensorboard"
        
        exp_name = getattr(exp_config, 'name', 'experiment')
        seed = getattr(exp_config, 'seed', 42)

        self.trial_name = f"{exp_name}::{seed}" 
        self.log_dir = self.log_dir / self.trial_name
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_train_end(self, task: 'Trainer') -> None:
        self.writer.close()

    def on_batch_end(self, task: 'Trainer') -> None:
        if task.cur_batch_cnt % self.log_interval != 0:
            return
        
        self.writer.add_scalar('Train/Loss', task.cur_batch_status.loss, task.cur_batch_cnt)

        if task.cur_val_status.eer < float('inf'):
            self.writer.add_scalar('Validation/Loss', task.cur_val_status.loss, task.cur_batch_cnt)
            self.writer.add_scalar('Validation/EER', task.cur_val_status.eer, task.cur_batch_cnt)
            
        if hasattr(task.optimizer, 'param_groups'):
            # TODO: now we only log the lr for the 1st set of params
            current_lr = task.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Training/LearningRate', current_lr, task.cur_batch_cnt)
        
        # Log best EER
        self.writer.add_scalar('Validation/BestEER', task.best_eer, task.cur_batch_cnt)
        
        self.writer.flush()