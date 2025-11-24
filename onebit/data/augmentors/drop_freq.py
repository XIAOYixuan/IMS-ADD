# encoding: utf-8
# author: Yixuan
#
#

from typing import Optional, Dict, Any
import numpy as np

from onebit.data.augmentors.registry import AugmentorRegistry
from onebit.data.augmentors.base import BaseAugmentor
from onebit.config import ConfigManager
from onebit.data.augmentors.helpers import notch_filter


@AugmentorRegistry.register('drop_freq')
class DropFreqAugmentation(BaseAugmentor):

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        data_conf = self.config_manager.get_data_config()
        drop_freq_conf = data_conf.get('aug', {}).get('drop_freq', {})
        
        self.probability = drop_freq_conf.get('probability', 0.5)
        self.sample_rate = drop_freq_conf.get('sample_rate', 16000)
        self.drop_freq_low = drop_freq_conf.get('drop_freq_low', 1e-14)
        self.drop_freq_high = drop_freq_conf.get('drop_freq_high', 1.0)
        self.drop_count_low = drop_freq_conf.get('drop_count_low', 1)
        self.drop_count_high = drop_freq_conf.get('drop_count_high', 2)
        self.drop_width = drop_freq_conf.get('drop_width', 0.05)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio
        
        drop_count = np.random.randint(self.drop_count_low, self.drop_count_high + 1)
        drop_range = self.drop_freq_high - self.drop_freq_low
        drop_frequencies = np.random.rand(drop_count) * drop_range + self.drop_freq_low

        # Filter parameters, hard coded just like speechbrain's impl
        filter_length = 101
        pad = filter_length // 2

        # create a delta filter 
        drop_filter = np.zeros(filter_length)
        drop_filter[pad] = 1 # impulse
        
        for frequency in drop_frequencies:
            notch_kernel = notch_filter(frequency, filter_length, self.drop_width)
            drop_filter = np.convolve(drop_filter, notch_kernel, mode='same')
        
        dropped_audio = np.convolve(audio, drop_filter, mode='same')
        
        return dropped_audio

if __name__ == '__main__':
    import sys
    import soundfile as sf
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python drop_freq.py <output_dir> [audio_file]")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    audio_file = sys.argv[2] if len(sys.argv) > 2 else "./tmp/test.flac"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = "onebit/configs/test.yaml"
    config_manager = ConfigManager(config_path)
    cli_args_dict = {
        "data.aug.drop_freq.probability": 1.0
    }
    config_manager.merge_with_cli(cli_args_dict)
    
    drop_freq_augmentor = DropFreqAugmentation(config_manager)
    
    from onebit.data import audio_util
    audio_array, sr = audio_util.get_audio(audio_file, trim_sil=False)
    augmented = drop_freq_augmentor(audio_array)
    
    output_path = output_dir / "drop_freq.wav"
    sf.write(str(output_path), augmented, sr)
    print(f"Augmented audio saved to {output_path}")

