# encoding: utf-8
# author: Yixuan
#
#
from typing import TYPE_CHECKING, Optional

from pathlib import Path
import torch

from onebit.config import ConfigManager
from onebit.task.callbacks.base import BaseCallback
from onebit.task.callbacks.registry import CallbackRegistry
from onebit.task.datatypes import ModelCheckpoint
from onebit.util import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..base import Task
    from ..trainer import Trainer
    from ..evaluator import Evaluator

@CallbackRegistry.register('load_checkpoints')
class LoadCheckPoints(BaseCallback):
    """
    Load checkpoints on task begin.
    
    If exp_config.ckpt_path is given, load the checkpoints from that path.
    Else:
        ckpt_path = self.root_dir/{exp_config.ckpt_tag}.pt
        exp_config.ckpt_tag default is 'best'
    
    Usage: 
        Config examples:
        1. Load specific checkpoint:
           exp:
             ckpt_path: "/path/to/checkpoint.pt"
             
        2. Load best checkpoint from exp_dir:
           exp:
             ckpt_tag: "best"  # or "last"
        
        Or CLI: python xxx.py --config exp_dir/train.yaml exp.ckpt_tag=best
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.checkpoint_path: Optional[Path] = None
        self._determine_checkpoint_path()

    def _determine_checkpoint_path(self) -> None:
        exp_config = self.config_manager.get_exp_config()

        # 1. specific ckpt 
        ckpt_path = getattr(exp_config, 'ckpt_path', None)
        if ckpt_path is not None:
            self.checkpoint_path = Path(ckpt_path)
            logger.info(f"using explicit ckpt path: {self.checkpoint_path}")
            return

        # 2. exp ckpt 
        if not hasattr(exp_config, 'ckpt_tag'):
            logger.info(f"no ckpt config can be found, skipping loading ckpt")
            return
        ckpt_tag = getattr(exp_config, 'ckpt_tag', 'best')
        
        root_path = getattr(exp_config, 'root_dir', None)
        if root_path is None:
            root_path = str(Path(__file__).parent.parent.parent.parent)
        
        self.root_dir = Path(root_path) / 'output'
        random_seed = str(exp_config.seed)
        exp_name = str(exp_config.name)
        exp_dir = self.root_dir / exp_name / random_seed
        self.checkpoint_path = exp_dir / f"{ckpt_tag}.pt"
        if not self.checkpoint_path.exists():
            logger.warning(f"Checkpoint file not found: {self.checkpoint_path}")
            self.checkpoint_path = None
        logger.info(f"using constructed checkpoint path: {self.checkpoint_path}")

    def on_task_begin(self, task: 'Task') -> None:
        if self.checkpoint_path is None:
            return
        if not hasattr(task, 'model'):
            raise ValueError(f"Task has no models!")
            
        try:
            device: torch.device = next(task.model.parameters()).device # type: ignore
            logger.info(f"loading checkpoint from: {self.checkpoint_path}")
            torch.serialization.add_safe_globals([ModelCheckpoint])
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, ModelCheckpoint):
                self._load_from_model_checkpoint(task, checkpoint)
            else:
                logger.error(f"Unknown checkpoint format: {type(checkpoint)}. Expected ModelCheckpoint dataclass.")
                return
                
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _load_from_model_checkpoint(self, task: 'Task', checkpoint: ModelCheckpoint) -> None:
        if checkpoint.model_state_dict is not None:
            task.model.load_state_dict(checkpoint.model_state_dict)  # type: ignore
            logger.info("Loaded full model checkpoint")
        elif checkpoint.backend_state_dict is not None:
            task.model.backend.load_state_dict(checkpoint.backend_state_dict)  # type: ignore
            logger.info("Loaded backend-only checkpoint")
        else:
            logger.info("All model state dicts are None, nothing loaded")

