# encoding: utf-8
# author: Yixuan
#
#
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from functools import partial
from transformers import AutoFeatureExtractor

from onebit.data.datasets.audiodataset import AudioSampleWithTensors
from onebit.data.collators.base import BaseAudioBatch, BaseCollator
from onebit.data.collators.registry import CollatorRegistry
from onebit.config import ConfigManager
from onebit.util import get_logger

logger = get_logger(__name__)

@dataclass
class AudioBatch(BaseAudioBatch):
    input_values: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    label_tensors: torch.Tensor
    labels: List[str] 
    uttids: List[str] 
    speakers: List[str] 
    attackers: List[str] 
    origin_ds: List[str] 
    audio_paths: List[str] 

@CollatorRegistry.register('audio')
class AudioCollator(BaseCollator):
    
    def __init__(self, config_manager: ConfigManager):
        data_config = config_manager.get_data_config().get('dataset')
        extractor_name = data_config.get('extractor_name', None)
        return_mask = data_config.get('return_mask', True)
        if extractor_name is None:
            # use the same extractor as the model
            extractor_name = config_manager.get_model_config().frontend.name
        logger.info(f"data extractor name: {extractor_name}")
        self.sample_rate = config_manager.get_sample_rate()
        
        self.fe = AutoFeatureExtractor.from_pretrained(extractor_name)
        self.fe.do_normalize = False
        self.fe.return_attention_mask = return_mask 

    def __call__(self, batch: List[AudioSampleWithTensors]) -> AudioBatch:
        audio_arrays  = [sample.audio_array for sample in batch]
        label_tensors = [sample.label_tensor for sample in batch]
        label_tensors = torch.stack(label_tensors)

        feats = self.fe(
            audio_arrays,
            sampling_rate=self.sample_rate,
            padding=True,                
            return_tensors="pt",
        )
        if hasattr(feats, 'attention_mask'):
            attention_mask = feats.attention_mask.bool()
        else:
            attention_mask = None

        return AudioBatch(
            input_values = feats.input_values,
            attention_mask = attention_mask, 
            label_tensors = label_tensors,
            labels = [s.label for s in batch],
            uttids = [s.uttid for s in batch],
            speakers = [s.speaker for s in batch],
            attackers = [s.attacker for s in batch],
            origin_ds = [s.origin_ds for s in batch],
            audio_paths = [s.audio_path for s in batch]
        ) 
