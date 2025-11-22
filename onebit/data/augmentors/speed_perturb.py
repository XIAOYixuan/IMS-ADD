# encoding: utf-8
# author: Yixuan
#
#

from typing import Optional, Dict, Any, List
import random
import librosa
import numpy as np

from onebit.data.augmentors.registry import AugmentorRegistry
from onebit.data.augmentors.base import BaseAugmentor
from onebit.config import ConfigManager


@AugmentorRegistry.register('speed_perturb')
class SpeedPerturbAugmentation(BaseAugmentor):
    """
    Apply speed perturbation augmentation to audio.
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        data_conf = self.config_manager.get_data_config()
        speed_conf = data_conf.get('aug', {}).get('speed_perturb', {})
        
        self.probability = speed_conf.get('probability', 0.5)
        self.sample_rate = speed_conf.get('sample_rate', 16000)
        self.speeds = speed_conf.get('speeds', [90, 100, 110])
        self.perturb_prob = speed_conf.get('perturb_prob', 1.0)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio
        
        speed = random.choice(self.speeds)
        new_freq = self.sample_rate * speed // 100
        resampled = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=new_freq)
        
        return resampled

if __name__ == '__main__':
    import sys
    import soundfile as sf
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python speed_perturb.py <output_dir> [audio_file]")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    audio_file = sys.argv[2] if len(sys.argv) > 2 else "./tmp/test.flac"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = "onebit/configs/test.yaml"
    config_manager = ConfigManager(config_path)
    cli_args_dict = {
        "data.aug.speed_perturb.probability": 1.0
    }
    config_manager.merge_with_cli(cli_args_dict)
    
    speed_perturb_augmentor = SpeedPerturbAugmentation(config_manager)
    
    from onebit.data import audio_util
    audio_array, sr = audio_util.get_audio(audio_file, trim_sil=False)
    augmented = speed_perturb_augmentor(audio_array)
    
    output_path = output_dir / "speed_perturb.wav"
    sf.write(str(output_path), augmented, sr)
    print(f"Augmented audio saved to {output_path}")

