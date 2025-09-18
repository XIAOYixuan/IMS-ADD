# encoding: utf-8
# author: Yixuan
#
#

from pathlib import Path
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from typing import Dict, List, Any
import pandas as pd
from dataclasses import dataclass

from onebit.config import ConfigManager
from onebit.data.datasets.base import BaseDataset
from onebit.data.datasets.registry import DatasetRegistry
from onebit.data.augmentors.augmentor import Augmentor
from onebit.data.postprocessors.postprocessor import PostProcessor
from onebit.data import audio_util
from onebit.util import get_logger
from onebit.data.datasets.audiodataset import AudioSample, AudioSampleWithTensors, AudioDataset
logger = get_logger(__name__)

@DatasetRegistry.register("nacaug")
class NACAugDataset(AudioDataset):

    def __init__(self, split, config_manager):
        super().__init__(split, config_manager)
        ds_conf = config_manager.get_data_config().get('dataset')
        self.probability = ds_conf.get('probability', 0.8)
        self.aug_rt = Path(ds_conf.get('aug_rt')) # augmentation dir
        self.aug_list = ds_conf.get('aug_list') # what nac should we use
        self.target = ds_conf.get('target', 'bonafide')
        logger.info(f'Available Augmentators: {self.aug_list}')

    def _replace_with_nac(self):
        pass

    def __getitem__(self, idx):
        sample: AudioSample = self.metadata[idx]

        prob = np.random.rand()
        if self.split == 'train' and prob < self.probability and sample.label=='bonafide':
            aug = np.random.choice(self.aug_list)
            target_path = self.aug_rt / aug / f'{sample.uttid}.wav'
            sample.audio_path = target_path
            sample.label = self.target
            # TODO: hard coded

        audio = self._load_audio(sample.audio_path)
        audio = self._process_audio(audio)
        label = self._encode_label(sample.label)
        #audio_tensor = torch.from_numpy(audio).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return AudioSampleWithTensors(
            uttid=sample.uttid,
            audio_path=sample.audio_path,
            origin_ds=sample.origin_ds,
            speaker=sample.speaker,
            attacker=sample.attacker,
            label=sample.label,
            audio_array=audio,
            label_tensor=label_tensor
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--split', '-s', type=str, default='train')
    args = parser.parse_args()

    config_manager = ConfigManager(args.config)
    ds = NACAugDataset(split=args.split, config_manager=config_manager)

    print(ds[15])
