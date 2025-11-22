# encoding: utf-8
# author: Yixuan
#
#

from typing import Optional, Dict, Any
import random
import torch
import torchaudio
import numpy as np

from onebit.data.augmentors.registry import AugmentorRegistry
from onebit.data.augmentors.base import BaseAugmentor
from onebit.config import ConfigManager


@AugmentorRegistry.register('codec')
class CodecAugmentation(BaseAugmentor):
    """
    Apply codec augmentation on mono audio.
    Implementation adapted from speechbrain.
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        data_conf = self.config_manager.get_data_config()
        codec_conf = data_conf.get('aug', {}).get('codec', {})
        
        self.probability = codec_conf.get('probability', 0.5)
        self.sample_rate = codec_conf.get('sample_rate', 16000)
        
        # Default formats: [("wav", "pcm_mulaw"), ("g722", None)]
        formats = codec_conf.get('formats', [("wav", "pcm_mulaw"), ("g722", None)])
        self.formats = formats

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio

        fmt, enc = random.choice(self.formats)

        x = torch.as_tensor(audio, dtype=torch.float32)
        x = x.unsqueeze(0).transpose(0, 1).cpu()

        eff = torchaudio.io.AudioEffector(format=fmt, encoder=enc)
        y = eff.apply(x, self.sample_rate).transpose(0, 1).squeeze(0)

        out = y.numpy()
        if np.issubdtype(audio.dtype, np.floating):
            out = out.astype(audio.dtype, copy=False)
        return out

if __name__ == '__main__':
    import sys
    import soundfile as sf
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python codec.py <output_dir> [audio_file]")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    audio_file = sys.argv[2] if len(sys.argv) > 2 else "./tmp/test.flac"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = "onebit/configs/test.yaml"
    config_manager = ConfigManager(config_path)
    cli_args_dict = {
        "data.aug.codec.probability": 1.0
    }
    config_manager.merge_with_cli(cli_args_dict)
    
    codec_augmentor = CodecAugmentation(config_manager)
    
    from onebit.data import audio_util
    audio_array, sr = audio_util.get_audio(audio_file, trim_sil=False)
    augmented = codec_augmentor(audio_array)
    
    output_path = output_dir / "codec.wav"
    sf.write(str(output_path), augmented, sr)
    print(f"Augmented audio saved to {output_path}")

