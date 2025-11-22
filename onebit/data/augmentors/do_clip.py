# encoding: utf-8
# author: Yixuan
#
#

from typing import Optional, Dict, Any
import numpy as np

from onebit.data.augmentors.registry import AugmentorRegistry
from onebit.data.augmentors.base import BaseAugmentor
from onebit.config import ConfigManager


@AugmentorRegistry.register('do_clip')
class DoClipAugmentation(BaseAugmentor):

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        data_conf = self.config_manager.get_data_config()
        clip_conf = data_conf.get('aug', {}).get('do_clip', {})
        
        self.probability = clip_conf.get('probability', 0.5)
        self.clip_low = clip_conf.get('clip_low', 0.5)
        self.clip_high = clip_conf.get('clip_high', 1.0)
        self.clip_prob = clip_conf.get('clip_prob', 1.0)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio
        
        clipping_range = self.clip_high - self.clip_low
        clip_value = np.random.rand() * clipping_range + self.clip_low
        clipped_audio = np.clip(audio, -clip_value, clip_value)
        
        return clipped_audio

if __name__ == '__main__':
    import sys
    import soundfile as sf
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python do_clip.py <output_dir> [audio_file]")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    audio_file = sys.argv[2] if len(sys.argv) > 2 else "./tmp/test.flac"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = "onebit/configs/test.yaml"
    config_manager = ConfigManager(config_path)
    cli_args_dict = {
        "data.aug.do_clip.probability": 1.0
    }
    config_manager.merge_with_cli(cli_args_dict)
    
    do_clip_augmentor = DoClipAugmentation(config_manager)
    
    from onebit.data import audio_util
    audio_array, sr = audio_util.get_audio(audio_file, trim_sil=False)
    augmented = do_clip_augmentor(audio_array)
    
    output_path = output_dir / "do_clip.wav"
    sf.write(str(output_path), augmented, sr)
    print(f"Augmented audio saved to {output_path}")

