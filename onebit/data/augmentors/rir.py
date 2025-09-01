# encoding: utf-8
# author: Yixuan
#
#

from typing import Optional, Dict, Any

from omegaconf import DictConfig 
from pathlib import Path
import torch
import numpy as np
import scipy

from onebit.data.augmentors.registry import AugmentorRegistry
from onebit.data.augmentors.base import BaseAugmentor
from onebit.config import ConfigManager
from onebit.data import audio_util

@AugmentorRegistry.register('rir')
class RIRAugmentation(BaseAugmentor):

    def __init__(self, 
                config_manager: ConfigManager):
        super().__init__(config_manager)
        data_conf = self.config_manager.get_data_config()
        rir_conf = data_conf.get('aug').get('rir')
        self.probability = rir_conf.get('probability', 0.3)
        # TODO: set by config if noise augmentation is needed in the future
        if rir_conf is None:
            raise ValueError(f'unknown rir config')
        rir_path = Path(rir_conf.rir_path)
        self.rir_files = self._get_all_rir_paths(rir_path)
        if not self.rir_files:
            raise ValueError(f'No RIR files found in {rir_path}')

    def _get_all_rir_paths(self, path):
        rir_files = []
        for rir_file in path.rglob("*.wav"):
            rir_files.append(rir_file)
        return rir_files

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        # audio shape: [length]
        prob = np.random.rand()
        if prob > self.probability:
            return audio
        self.audio_power = float((audio**2).mean())
        if self.audio_power < 1e-10:
            return audio  # Skip augmentation for silent audio
        rir_path = np.random.choice(self.rir_files)
        rir, sample_rate = audio_util.get_audio(rir_path, to_mono=True, trim_sil=False)
        
        augmented = scipy.signal.convolve(audio, rir, 
                        mode="full")[:audio.shape[0]]
        
        # restore original power level
        augment_power = float((augmented**2).mean())
        if augment_power > 1e-10:
            scale = float(np.sqrt(self.audio_power / augment_power))
            augmented = scale * augmented

        return augmented

if __name__ == '__main__':
    config_path = "onebit/configs/test.yaml"
    config_manager = ConfigManager(config_path)
    cli_args_dict = {
        "data.aug.rir.probability": 1.0,
        "data.aug.rir.rir_path": "./tmp/RIRS_NOISES/"
    }
    config_manager.merge_with_cli(cli_args_dict)

    rir_augmentor = RIRAugmentation(config_manager)

    audio_file = "./tmp/test.flac"
    audio_array, sr = audio_util.get_audio(audio_file, trim_sil=False)
    augmented = rir_augmentor(audio_array)

    import soundfile as sf
    sf.write("test_augmented.wav", augmented, sr)
