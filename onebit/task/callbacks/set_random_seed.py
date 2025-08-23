# encoding: utf-8
# author: Yixuan
#
#
from typing import TYPE_CHECKING

import random
import numpy as np
import torch

from onebit.config import ConfigManager
from onebit.task.callbacks.base import BaseCallback
from onebit.task.callbacks.registry import CallbackRegistry
from onebit.util import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..base import Task

@CallbackRegistry.register('set_random_seed')
class SetRandomSeed(BaseCallback):
    """
    On task begin, check if exp.seed is set in config.
    If not set, automatically generate a random seed.
    Set the random seed for all modules.
    Then update the config: 
        self.config_manager.config.exp.seed=seed
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        
        if hasattr(self.config_manager.config.exp, 'seed') and self.config_manager.config.exp.seed is not None:
            seed = self.config_manager.config.exp.seed
            logger.info(f"Using existing seed from config: {seed}")
        else:
            seed = random.randint(1, 1000000)
            self.config_manager.config.exp.seed = seed
            logger.info(f"Generated new random seed: {seed}")
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"Set random seed to {seed}")