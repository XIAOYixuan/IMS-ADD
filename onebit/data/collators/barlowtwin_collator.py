# encoding: utf-8
# author: Yixuan
#
#

import torch
from typing import List

from onebit.data.datasets.barlow_twin_dataset import BarlowTwinSampleWithTensors
from onebit.data.collators.audiocollator import AudioCollator, AudioBatch
from onebit.data.collators.registry import CollatorRegistry
from onebit.config import ConfigManager


@CollatorRegistry.register('barlow_twin')
class BarlowTwinCollator(AudioCollator):
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)

    def __call__(self, batch: List[BarlowTwinSampleWithTensors]) -> AudioBatch:
        """
        input_values[0:k] = augmented ver 1
        input_values[k:2k] = augmented ver 2 
        k = batch_size
        """
        # Collect primary and twin audio arrays
        primary_audio_arrays = [sample.audio_array for sample in batch]
        twin_audio_arrays = [sample.twin_audio_array for sample in batch]
        
        # Combine primary and twin audio arrays
        all_audio_arrays = primary_audio_arrays + twin_audio_arrays
        
        # Collect label tensors and duplicate them
        label_tensors = [sample.label_tensor for sample in batch]
        all_label_tensors = label_tensors + label_tensors  # Duplicate for twins
        all_label_tensors = torch.stack(all_label_tensors)
        
        # Process audio through feature extractor
        feats = self.fe(
            all_audio_arrays,
            sampling_rate=self.sample_rate,
            padding=True,                
            return_tensors="pt",
        )
        
        # Duplicate all metadata fields
        labels = [s.label for s in batch] * 2
        uttids = [s.uttid for s in batch] * 2
        speakers = [s.speaker for s in batch] * 2
        attackers = [s.attacker for s in batch] * 2
        origin_ds = [s.origin_ds for s in batch] * 2
        audio_paths = [s.audio_path for s in batch] * 2

        return AudioBatch(
            input_values=feats.input_values,
            attention_mask=feats.attention_mask.bool(),  
            label_tensors=all_label_tensors,
            labels=labels,
            uttids=uttids,
            speakers=speakers,
            attackers=attackers,
            origin_ds=origin_ds,
            audio_paths=audio_paths
        )
