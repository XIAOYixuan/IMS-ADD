# encoding: utf-8
# author: Yixuan
#
#

from pathlib import Path
import numpy as np
import logging

from onebit.data.augmentors.registry import AugmentorRegistry
from onebit.data.augmentors.base import BaseAugmentor
from onebit.config import ConfigManager
from onebit.data.augmentors.helpers import select_audio_from_df, load_csv_dataframe, align_waveform, compute_amplitude, dB_to_amplitude

logger = logging.getLogger(__name__)

@AugmentorRegistry.register('add_noise')
class AddNoiseAugmentation(BaseAugmentor):
    """
    Adpated from speechbrain's impl
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        data_conf = self.config_manager.get_data_config()
        noise_conf = data_conf.get('aug', {}).get('add_noise', {})
        
        self.probability = noise_conf.get('probability', 0.5)
        self.sample_rate = noise_conf.get('sample_rate', 16000)
        self.snr_low = noise_conf.get('snr_low', 5.0)
        self.snr_high = noise_conf.get('snr_high', 20.0)
        self.pad_noise = noise_conf.get('pad_noise', False)
        self.normalize = noise_conf.get('normalize', False)
        
        noise_csv = noise_conf.get('noise_csv', None)
        if noise_csv is None:
            raise ValueError("add_noise augmentation requires 'noise_csv' config parameter")
        noise_csv_path = Path(noise_csv)
        if not noise_csv_path.exists():
            raise ValueError(f"noise CSV file not found: {noise_csv_path}")
        self.noise_df = load_csv_dataframe(str(noise_csv_path))

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio
        
        audio_power = float((audio**2).mean())
        if audio_power < 1e-10:
            logger.warning("Audio power is too low, likely a silent audio, returning original audio")
            return audio
        
        SNR = np.random.uniform(self.snr_low, self.snr_high)
        clean_amplitude = compute_amplitude(audio) 
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude
        noisy_waveform = audio * (1 - noise_amplitude_factor)

        noise = select_audio_from_df(self.noise_df, self.sample_rate)
        audio_len = len(audio)
        noise = align_waveform(noise, audio_len, self.pad_noise, None)
        
        noise_amplitude = compute_amplitude(noise)
        noise_waveform = noise * new_noise_amplitude / (noise_amplitude + 1e-14)
        noisy_waveform += noise_waveform

        return noisy_waveform

if __name__ == '__main__':
    import sys
    import soundfile as sf
    from pathlib import Path
    
    if len(sys.argv) < 3:
        print("Usage: python add_noise.py <output_dir> <noise_csv> [audio_file]")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    noise_csv = sys.argv[2]
    audio_file = sys.argv[3] if len(sys.argv) > 3 else "./tmp/test.flac"
    
    if not Path(noise_csv).exists():
        raise ValueError(f"noise CSV file not found: {noise_csv}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = "onebit/configs/test.yaml"
    config_manager = ConfigManager(config_path)
    cli_args_dict = {
        "data.aug.add_noise.probability": 1.0,
        "data.aug.add_noise.noise_csv": noise_csv
    }
    config_manager.merge_with_cli(cli_args_dict)
    
    add_noise_augmentor = AddNoiseAugmentation(config_manager)
    
    from onebit.data import audio_util
    audio_array, sr = audio_util.get_audio(audio_file, trim_sil=False)
    augmented = add_noise_augmentor(audio_array)
    
    output_path = output_dir / "add_noise.wav"
    sf.write(str(output_path), augmented, sr)
    print(f"Augmented audio saved to {output_path}")

