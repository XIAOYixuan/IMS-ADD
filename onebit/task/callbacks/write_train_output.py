# encoding: utf-8
# author: Yixuan
#
#

from pathlib import Path
import json 
import time
from typing import TYPE_CHECKING, Dict, Any, List, Optional

import torch

from onebit.config import ConfigManager
from onebit.task.callbacks.base import BaseCallback
from onebit.task.callbacks.registry import CallbackRegistry
from onebit.task.datatypes import ModelCheckpoint, TrainStatus, ValStatus
from onebit.util import get_logger
logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..trainer import Trainer

@CallbackRegistry.register('write_train_output')
class WriteTrainOutputCallBack(BaseCallback):
    """
    After every log interval, log the output, 
        if cur eer < best eer, save the checkpoint
    After each epoch, save the checkpoint
    At the beginning of the task, save the config
    At the end of the task, save the summary info
    """

    def __init__(self, config_manager):
        super().__init__(config_manager)

        exp_config = self.config_manager.get_exp_config()
        random_seed = str(exp_config.seed)
        exp_name = str(exp_config.name)

        root_path = getattr(exp_config, 'root_dir', None) 
        if root_path is None:
            root_path = str(Path(__file__).parent.parent.parent.parent)
        self.root_dir = Path(root_path)/'output'
        self.exp_dir = self.root_dir / exp_name / random_seed
        
        # Check if save_last is configured in callbacks.write_train_output
        self.save_last = True  # default to True
        if hasattr(exp_config, 'callbacks') and hasattr(exp_config.callbacks, 'write_train_output'):
            self.save_last = getattr(exp_config.callbacks.write_train_output, 'save_last', True)
        
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f'{__name__} initialized, output to {self.exp_dir}')

    def on_train_begin(self, task: 'Trainer') -> None:
        config_path = self.exp_dir / 'train.yaml'
        logger.info(f"saving config to {config_path}")
        self.config_manager.save_config(config_path)
        self.config_manager.print_config()

        logger.info("Model architecture:")
        logger.info(task.model)

        # log time
        self.start_time = time.time()

    def on_train_end(self, task: 'Trainer') -> None:
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.exp_info = {
            'total_epochs': task.cur_epoch,
            'total_batches': task.cur_batch_cnt,
            'best_eer': task.best_eer,
            'duration_formatted': time.strftime('%H:%M:%S', time.gmtime(duration)),  
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)),  
            'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time)),
        }

        exp_info_path = self.exp_dir / 'train_summary.json'
        with open(exp_info_path, 'w') as f:
            json.dump(self.exp_info, f, indent=2)
        
        logger.info(f"Training info saved to {exp_info_path}")

    def save_checkpoint(self, task: 'Trainer', filename: Optional[str] = None) -> None:
        if filename is None:
            filename = f"checkpoint_epoch_{task.cur_epoch}.pt"
        elif not filename.endswith('.pt'):
            filename += '.pt'
        
        checkpoint_path = self.exp_dir / filename
        
        # Check if frontend is frozen
        frontend_config = self.config_manager.get_model_config().frontend
        is_frontend_frozen = getattr(frontend_config, 'freeze_frontend', True)
        
        if is_frontend_frozen:
            # only store the backend
            backend_state_dict = task.model.backend.state_dict()
            checkpoint = ModelCheckpoint(
                epoch=task.cur_epoch,
                batch_count=task.cur_batch_cnt,
                best_eer=task.best_eer,
                frontend_frozen=is_frontend_frozen,
                model_state_dict=None,
                backend_state_dict=backend_state_dict
            )
            logger.info(f"saving ckpt (backend only) to {checkpoint_path}")
        else:
            full_state_dict = task.model.state_dict()
            checkpoint = ModelCheckpoint(
                epoch=task.cur_epoch,
                batch_count=task.cur_batch_cnt,
                best_eer=task.best_eer,
                frontend_frozen=is_frontend_frozen,
                model_state_dict=full_state_dict,
                backend_state_dict=None
            )
            logger.info(f"saving ckpt (full) to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

    def on_epoch_end(self, task: 'Trainer') -> None:
        if self.save_last:
            self.save_checkpoint(task, 'last')

    def on_batch_end(self, task: 'Trainer') -> None:
        cur_stat: ValStatus = task.cur_val_status
        if cur_stat.eer < task.best_eer:
            task.best_eer = cur_stat.eer
            self.save_checkpoint(task, 'best')
