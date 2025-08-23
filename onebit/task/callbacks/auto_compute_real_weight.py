# encoding: utf-8
# author: Yixuan
#
#
from typing import TYPE_CHECKING

from pathlib import Path
import pandas as pd
import json

from onebit.config import ConfigManager
from onebit.task.callbacks.base import BaseCallback
from onebit.task.callbacks.registry import CallbackRegistry
from onebit.util import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..base import Task

@CallbackRegistry.register('auto_compute_real_weight')
class AutoComputeRealWeight(BaseCallback):

    """
    This class automatically compute the real_weight that
    will be used any BCE based loss from the dataset.
    It has higher priority than the real_weights defined
    in config.
    Once the real_weight is computed, it'll modify the 
    config yaml, so the stored yaml reflects the actual
    real_weight
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        callbacks_config = self.config_manager.get_exp_config().callbacks
        my_config = getattr(callbacks_config, 'auto_compute_real_weight', None) if callbacks_config else None

    def _encode_label(self, label: str) -> int:
        """Encode label to integer (same logic as AudioDataset)"""
        if label.lower() == 'bonafide':
            return 1
        elif label.lower() == 'spoof':
            return 0
        else:
            raise ValueError(f'Invalid label: [{label}], must be [bonafide, spoof]')

    def on_task_begin(self, task: 'Task') -> None:
        base_path = self.config_manager.get_dataset_path()
        dataset_dir = Path(base_path)
        
        train_txt_file = dataset_dir / "train.txt"
        if not train_txt_file.exists():
            logger.info(f"train.txt not found at {train_txt_file}, won't set the real_weight")
        
        txt_data = pd.read_csv(train_txt_file, sep='\t', header=None, 
                              names=['uttid', 'origin_ds', 'speaker', 'attacker', 'label'])
        
        num_real = 0
        num_fake = 0
        for _, row in txt_data.iterrows():
            label = str(row['label'])
            encoded_label = self._encode_label(label)
            if encoded_label == 1:  
                num_real += 1
            else:  
                num_fake += 1
        
        # Handle edge cases
        if num_fake == 0 or num_real == 0:
            raise ValueError(f"no bonafide or spoof found in the dataset! num_fake={num_fake}, num_real={num_real}")
        else:
            real_weight = float(num_fake) / float(num_real)
        
        # update the config, will be auto saved by write_xx_output cb
        self.config_manager.config.loss.real_weight = real_weight
        return