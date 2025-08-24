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

from onebit.data.audiodataset import AudioSampleWithTensors
from onebit.config import ConfigManager


@dataclass
class AudioBatch:
    input_values: torch.Tensor
    attention_mask: torch.Tensor
    label_tensors: torch.Tensor
    labels: List[str] 
    uttids: List[str] 
    speakers: List[str] 
    attackers: List[str] 
    origin_ds: List[str] 
    audio_paths: List[str] 
    
    def to(self, device: torch.device, non_blocking: bool = True):
        for fld, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, fld, value.to(device, non_blocking=non_blocking))
        return self

    def asdict(self) -> Dict[str, Any]:
        return self.__dict__


class AudioCollator:
    
    def __init__(self, config_manager: ConfigManager):
        model_config = config_manager.get_model_config()
        extractor_name = model_config.frontend.name
        
        self.sample_rate = config_manager.get_sample_rate()
        
        self.fe = AutoFeatureExtractor.from_pretrained(extractor_name)
        self.fe.do_normalize = False
        self.fe.return_attention_mask = True

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

        return AudioBatch(
            input_values = feats.input_values,
            attention_mask = feats.attention_mask.bool(),  
            label_tensors = label_tensors,
            labels = [s.label for s in batch],
            uttids = [s.uttid for s in batch],
            speakers = [s.speaker for s in batch],
            attackers = [s.attacker for s in batch],
            origin_ds = [s.origin_ds for s in batch],
            audio_paths = [s.audio_path for s in batch]
        ) 
