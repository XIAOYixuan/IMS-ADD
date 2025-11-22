# encoding: utf-8
# author: Yixuan
#
#

from typing import Optional, Dict, Any
import numpy as np

from onebit.data.augmentors.registry import AugmentorRegistry
from onebit.data.augmentors.base import BaseAugmentor
from onebit.config import ConfigManager
from onebit.data.augmentors.helpers import compute_amplitude


@AugmentorRegistry.register('drop_chunk')
class DropChunkAugmentation(BaseAugmentor):

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        data_conf = self.config_manager.get_data_config()
        drop_chunk_conf = data_conf.get('aug', {}).get('drop_chunk', {})
        
        self.probability = drop_chunk_conf.get('probability', 0.5)
        self.drop_length_low = drop_chunk_conf.get('drop_length_low', 100)
        self.drop_length_high = drop_chunk_conf.get('drop_length_high', 1000)
        self.drop_count_low = drop_chunk_conf.get('drop_count_low', 1)
        self.drop_count_high = drop_chunk_conf.get('drop_count_high', 10)
        self.drop_start = drop_chunk_conf.get('drop_start', 0)
        self.drop_end = drop_chunk_conf.get('drop_end', None)
        self.noise_factor = drop_chunk_conf.get('noise_factor', 0.0)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio
        
        audio_len = len(audio)
        dropped_audio = audio.copy()
        
        clean_amplitude = compute_amplitude(audio)
        
        drop_times = np.random.randint(self.drop_count_low, self.drop_count_high + 1)
        
        for _ in range(drop_times):
            drop_length = np.random.randint(self.drop_length_low, self.drop_length_high + 1)
            
            start_min = self.drop_start
            if start_min < 0:
                start_min += audio_len
            
            start_max = self.drop_end if self.drop_end is not None else audio_len
            if start_max < 0:
                start_max += audio_len
            start_max = max(0, start_max - drop_length)
            
            if start_min >= start_max:
                continue
                
            start = np.random.randint(start_min, start_max + 1)
            end = min(start + drop_length, audio_len)
            
            if self.noise_factor == 0.0:
                dropped_audio[start:end] = 0.0
            else:
                noise_max = 2 * clean_amplitude * self.noise_factor
                noise_vec = np.random.rand(end - start) * 2 * noise_max - noise_max
                dropped_audio[start:end] = noise_vec
        
        return dropped_audio

if __name__ == '__main__':
    import sys
    import soundfile as sf
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python drop_chunk.py <output_dir> [audio_file]")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    audio_file = sys.argv[2] if len(sys.argv) > 2 else "./tmp/test.flac"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = "onebit/configs/test.yaml"
    config_manager = ConfigManager(config_path)
    cli_args_dict = {
        "data.aug.drop_chunk.probability": 1.0
    }
    config_manager.merge_with_cli(cli_args_dict)
    
    drop_chunk_augmentor = DropChunkAugmentation(config_manager)
    
    from onebit.data import audio_util
    audio_array, sr = audio_util.get_audio(audio_file, trim_sil=False)
    augmented = drop_chunk_augmentor(audio_array)
    
    output_path = output_dir / "drop_chunk.wav"
    sf.write(str(output_path), augmented, sr)
    print(f"Augmented audio saved to {output_path}")

