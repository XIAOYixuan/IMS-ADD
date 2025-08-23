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
    from ..evaluator import Evaluator

@CallbackRegistry.register('write_infer_output')
class WriteInferOutputCallback(BaseCallback):
    """
    at the end of infer
    write the uttid, prediction, label to exp_dir/infer/
    the name is the configs data.dataset.name
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        
        exp_config = self.config_manager.get_exp_config()
        random_seed = str(exp_config.seed)
        exp_name = str(exp_config.name)

        root_path = getattr(exp_config, 'root_dir', None)
        if root_path is None:
            root_path = str(Path(__file__).parent.parent.parent.parent)
        
        self.root_dir = Path(root_path)/'output'
        self.exp_dir = self.root_dir / exp_name / random_seed
        self.infer_dir = self.exp_dir / 'infer'
        
        self.infer_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_name = self.config_manager.get_dataset_name()
        
        logger.info(f'{__name__} initialized, will output to {self.infer_dir}')

    def on_infer_begin(self, task: 'Evaluator') -> None:
        logger.info("Model architecture:")
        logger.info(task.model)

    def on_infer_end(self, task: 'Evaluator') -> None:
        pred_tensor = task.pred_tensor
        targ_tensor = task.targ_tensor
        uttids = task.uttids
        eer = task.eer
        thresh = task.thresh
        
        predictions = pred_tensor.tolist()
        labels = targ_tensor.tolist()
        
        results_df = pd.DataFrame({
            'uttid': uttids,
            'prediction': predictions,
            'label': labels
        })
        
        output_file = self.infer_dir / f"{self.dataset_name}.csv"
        
        results_df.to_csv(output_file, index=False)
        
        summary_file = self.infer_dir / f"{self.dataset_name}_summary.json"
        summary_data = {
            "dataset": self.dataset_name,
            "total_samples": len(uttids),
            "equal_error_rate": float(eer),
            "threshold": float(thresh)
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"infer results written to {output_file}")
        logger.info(f"summary written to {summary_file}")
        logger.info(f"eer: {eer:.4f}, thresh: {thresh:.4f}")