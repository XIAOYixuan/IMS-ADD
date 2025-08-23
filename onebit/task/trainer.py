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
from onebit.data import AudioDataset, AudioCollator, AudioBatch
from onebit.model.datatypes import BackendOutput
from onebit.task.callbacks.base import BaseCallback
from onebit.loss.datatypes import LossOutput
from onebit.task.base import Task
from onebit.task.datatypes import TrainStatus, ValStatus 
from onebit.model.model import Model
from onebit.loss.factory import LossFactory
from onebit.loss.base import BaseLoss
from onebit.metrics.eer import calculate_eer
from onebit.util import get_logger

logger = get_logger(__name__)


class Trainer(Task):

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.builtin_callback_names = [
            'set_random_seed',
            'lr_scheduler', 
            'write_train_output',
            'tensorboard_monitor']
        self.callbacks = self._create_callbacks()
        for cb in self.callbacks:
            cb.on_task_begin(self) 

        # model, loss, optimizer
        self.model: Model = Model(config_manager)
        self.loss_fn: BaseLoss = LossFactory.create(config_manager)
        self.optimizer: torch.optim.Optimizer = self._create_optimizer()

        # train stat
        self.cur_epoch = 0
        self.cur_batch_cnt = 0
        self.cur_epoch_status: TrainStatus = TrainStatus()
        self.cur_batch_status: TrainStatus = TrainStatus()
        self.cur_val_status: ValStatus= ValStatus()
        self.best_eer = float('inf')
        exp_config = self.config_manager.get_exp_config()
        self.val_interval = getattr(exp_config, 'val_interval', None)

        # for early stopping
        self.should_stop = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss_fn.to(self.device)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        exp_config = self.config_manager.get_exp_config()
        opt_name = getattr(exp_config.optimizer, 'name', 'adam')
        lr = getattr(exp_config, 'lr', 0.0001)

        if opt_name.lower() == 'adam':
            beta1 = getattr(exp_config.optimizer, 'beta1', 0.9)
            beta2 = getattr(exp_config.optimizer, 'beta', 0.999)
            return torch.optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2))
        else:
            raise ValueError(f"unk optimizer: {opt_name}")
    
    def run_one_epoch(self) -> TrainStatus:
        self.model.train()
        epoch_status = TrainStatus()
        num_batches = 0

        for batch_idx, audio_batch in enumerate(tqdm(self.trainloader)):
            self.cur_batch_cnt += 1
            audio_batch = audio_batch.to(self.device) # type: AudioBatch

            for cb in self.callbacks:
                cb.on_batch_begin(self)

            self.optimizer.zero_grad()
            backend_output: BackendOutput = self.model(audio_batch)
            loss_output: LossOutput = self.loss_fn(audio_batch, backend_output)
            loss = loss_output.loss 
            loss.backward()
            self.optimizer.step()

            self.cur_batch_status.loss = loss.item()
            epoch_status.loss += self.cur_batch_status.loss
            num_batches += 1

            if self.val_interval and self.cur_batch_cnt % self.val_interval == 0:
                self.cur_val_status = self.validate()

            for cb in self.callbacks:
                cb.on_batch_end(self)

        if num_batches > 0:
            epoch_status.loss /= num_batches
        return epoch_status

    def validate(self) -> ValStatus:
        self.model.eval()
        cur_status = ValStatus()
        num_batches = 0
        all_pred = []
        all_targ = []

        with torch.no_grad():
            for audio_batch in tqdm(self.devloader):
                audio_batch = audio_batch.to(self.device) # type: AudioBatch
                model_out: BackendOutput = self.model(audio_batch)
                loss_out: LossOutput = self.loss_fn(audio_batch, model_out)

                cur_status.loss += loss_out.loss.item()
                num_batches += 1

                pred = model_out.predictions.cpu()
                labels = audio_batch.label_tensors.cpu()
                all_pred.append(pred)
                all_targ.append(labels)
        
        pred_tensor: torch.Tensor = torch.cat(all_pred)
        targ_tensor: torch.Tensor = torch.cat(all_targ) 
        eer, thresh = calculate_eer(pred_tensor, targ_tensor)
        avg_loss = cur_status.loss / num_batches if num_batches > 0 else 0.0
        
        cur_status.loss = avg_loss
        cur_status.eer = eer
        cur_status.thresh = thresh
        
        return cur_status

    def start(self):
        train_key = 'train'
        dev_key = 'dev'
        trainset = AudioDataset(train_key, self.config_manager)
        devset = AudioDataset(dev_key, self.config_manager)
        collator = AudioCollator(self.config_manager)
        self.trainset = trainset
        self.devset = devset
        logger.info(f"Training set size: {len(trainset)} samples")
        logger.info(f"Development set size: {len(devset)} samples")

        loader_config = self.config_manager.get_data_config().loader # type: ignore
        trainloader = DataLoader(
            trainset, 
            batch_size=loader_config.bz,
            shuffle=loader_config.shuffle,
            collate_fn=collator,
            num_workers=loader_config.train_nw
        )
        devloader = DataLoader(
            devset,
            batch_size=loader_config.val_bz,
            shuffle=False,
            collate_fn=collator,
            num_workers=loader_config.val_nw
        )
        self.trainloader = trainloader
        self.devloader = devloader

        exp_config = self.config_manager.get_exp_config()
        num_epochs = getattr(exp_config, 'num_epochs', 100)

        for cb in self.callbacks:
            cb.on_train_begin(self)

        logger.info(f"start training for {num_epochs} epochs")
        try:
            for epoch in range(1, num_epochs+1):
                self.cur_epoch = epoch

                for cb in self.callbacks:
                    cb.on_epoch_begin(self)

                self.cur_epoch_status = self.run_one_epoch()
                self.cur_val_status = self.validate()
                logger.info(f'Epoch {epoch}: cur epoch status: {self.cur_epoch_status}')
                logger.info(f'Epoch {epoch}: cur val status: {self.cur_val_status}')

                for callback in self.callbacks:
                    callback.on_epoch_end(self)
                
                if self.should_stop:
                    logger.info("early stopping triggered")
                    break
        except KeyboardInterrupt:
            logger.info("training interrupted by user")
        except Exception as e:
            logger.error(f"training failed with error: {e}")
            raise 
        finally:
            for cb in self.callbacks:
                cb.on_train_end(self)
            for cb in self.callbacks:
                cb.on_task_end(self)