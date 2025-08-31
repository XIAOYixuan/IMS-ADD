# encoding: utf-8
# author: Yixuan
#
#

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from onebit.config import ConfigManager
from onebit.data import DatasetFactory, CollatorFactory, AudioBatch
from onebit.model.datatypes import FrontendOutput, BackendOutput
from onebit.loss.datatypes import LossOutput
from onebit.task.base import Task
from onebit.task.datatypes import TrainStatus, ValStatus 
from onebit.task.callbacks.base import BaseCallback
from onebit.task.callbacks.factory import CallbackFactory
from onebit.model.model import Model
from onebit.loss.factory import LossFactory
from onebit.loss.base import BaseLoss
from onebit.metrics.eer import calculate_eer
from onebit.util import get_logger

logger = get_logger(__name__)

class Evaluator(Task):

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.builtin_callback_names = [
            'set_random_seed',
            'load_checkpoints', 
            'write_infer_output']
        self.callbacks = self._create_callbacks()
        
        # model, loss, optimizer
        self.model: Model = Model(config_manager)
        self.loss_fn: BaseLoss = LossFactory.create(config_manager)
 
        for cb in self.callbacks:
            cb.on_task_begin(self)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss_fn.to(self.device)

    def start(self):
        val_key = self.config_manager.get_data_config().val_key
        val_set = DatasetFactory.create(val_key, self.config_manager)
        collator = CollatorFactory.create(self.config_manager)

        loader_config = self.config_manager.get_data_config().loader # type: ignore
        self.val_dataloader = DataLoader(
            val_set,
            batch_size=loader_config.test_bz,
            shuffle=False,
            collate_fn=collator,
            num_workers=loader_config.test_nw
        )

        for cb in self.callbacks:
            cb.on_infer_begin(self)

        logger.info(f"Start evaluation on {val_key} set, in total {len(val_set)} samples.")

        all_pred = [] 
        all_targ = []
        all_utt = []
        self.model.eval()
        with torch.no_grad():
            for audio_batch in tqdm(self.val_dataloader):
                audio_batch = audio_batch.to(self.device) # type: AudioBatch
                model_out: BackendOutput = self.model(audio_batch)

                pred = model_out.predictions.cpu()
                labels = audio_batch.label_tensors.cpu()
                all_pred.append(pred)
                all_targ.append(labels)
                all_utt.extend(audio_batch.uttids)
        
        pred_tensor: torch.Tensor = torch.cat(all_pred)
        targ_tensor: torch.Tensor = torch.cat(all_targ) 
        eer, thresh = calculate_eer(pred_tensor, targ_tensor)

        self.eer = eer
        self.thresh = thresh
        self.pred_tensor = pred_tensor
        self.targ_tensor = targ_tensor
        self.uttids = all_utt

        for cb in self.callbacks:
            cb.on_infer_end(self)