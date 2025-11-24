# encoding: utf-8
# author: Yixuan
#
#

from pathlib import Path
import numpy as np

from onebit.data.augmentors.registry import AugmentorRegistry
from onebit.data.augmentors.base import BaseAugmentor
from onebit.config import ConfigManager
from onebit.data.augmentors.helpers import select_audio_from_df, load_csv_dataframe


@AugmentorRegistry.register('morph')
class MorphAugmentation(BaseAugmentor):
    """
    Similar logic to add_noise, the adapted from ESPNet's impl
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        data_conf = self.config_manager.get_data_config()
        morph_conf = data_conf.get('aug', {}).get('morph', {})
        
        self.probability = morph_conf.get('probability', 0.5)
        self.sample_rate = morph_conf.get('sample_rate', 16000)
        self.noise_db_low = morph_conf.get('noise_db_low', 5.0)
        self.noise_db_high = morph_conf.get('noise_db_high', 20.0)
        
        noise_csv = morph_conf.get('noise_csv', None)
        if noise_csv is None:
            raise ValueError("morph augmentation requires 'noise_csv' config parameter")
        noise_csv_path = Path(noise_csv)
        if not noise_csv_path.exists():
            raise ValueError(f"noise CSV file not found: {noise_csv_path}")
        self.noise_df = load_csv_dataframe(str(noise_csv_path))

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.probability:
            return audio
            
        noise = select_audio_from_df(self.noise_df, self.sample_rate)
        
        audio_power = float((audio**2).mean())
        noise_db = np.random.uniform(self.noise_db_low, self.noise_db_high)

        audio_nsamples = audio.shape[0]
        noise_nsamples = noise.shape[0]

        # align noise and audio such that they have the same length
        if audio_nsamples == noise_nsamples:
            pass
        elif audio_nsamples > noise_nsamples:
            offset = np.random.randint(0, audio_nsamples - noise_nsamples)
            noise = np.pad(
                noise,
                (offset, audio_nsamples - noise_nsamples - offset),
                mode="wrap",
            )
        else:
            offset = np.random.randint(0, noise_nsamples - audio_nsamples)
            noise = noise[offset : offset + audio_nsamples]

        noise_power = float((noise**2).mean())
        scale = (
            10 ** (-noise_db / 20)
            * np.sqrt(audio_power)
            / np.sqrt(np.maximum(noise_power, 1e-10))
        )
        audio = audio + scale * noise
        return audio

if __name__ == '__main__':
    import sys
    import soundfile as sf
    from pathlib import Path
    
    if len(sys.argv) < 3:
        print("Usage: python morph.py <output_dir> <noise_csv> [audio_file]")
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
        "data.aug.morph.probability": 1.0,
        "data.aug.morph.noise_csv": noise_csv
    }
    config_manager.merge_with_cli(cli_args_dict)
    
    morph_augmentor = MorphAugmentation(config_manager)
    
    from onebit.data import audio_util
    audio_array, sr = audio_util.get_audio(audio_file, trim_sil=False)
    augmented = morph_augmentor(audio_array)
    
    output_path = output_dir / "morph.wav"
    sf.write(str(output_path), augmented, sr)
    print(f"Augmented audio saved to {output_path}")

