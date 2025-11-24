# encoding: utf-8
# author: Yixuan
#
#

from pathlib import Path
import numpy as np

from onebit.data.augmentors.registry import AugmentorRegistry
from onebit.data.augmentors.base import BaseAugmentor
from onebit.config import ConfigManager
from onebit.data.augmentors.helpers import select_multiple_audio_from_df, load_csv_dataframe, align_waveform, compute_amplitude, dB_to_amplitude


@AugmentorRegistry.register('add_babble')
class AddBabbleAugmentation(BaseAugmentor):
    """
    Apply babble augmentation to audio.
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        data_conf = self.config_manager.get_data_config()
        babble_conf = data_conf.get('aug', {}).get('add_babble', {})
        
        self.probability = babble_conf.get('probability', 0.5)
        self.sample_rate = babble_conf.get('sample_rate', 16000)
        self.speaker_count = babble_conf.get('speaker_count', 3)
        self.snr_low = babble_conf.get('snr_low', 0.0)
        self.snr_high = babble_conf.get('snr_high', 0.0)
        self.pad_noise = babble_conf.get('pad_noise', False)
        
        babble_csv = babble_conf.get('babble_csv', None)
        if babble_csv is None:
            raise ValueError("add_babble augmentation requires 'babble_csv' config parameter")
        babble_csv_path = Path(babble_csv)
        if not babble_csv_path.exists():
            raise ValueError(f"babble CSV file not found: {babble_csv_path}")
        self.babble_df = load_csv_dataframe(str(babble_csv_path))

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() > self.probability:
            return audio
        
        babble_waveforms = select_multiple_audio_from_df(
            self.babble_df, 
            self.speaker_count, 
            self.sample_rate)
        
        SNR = np.random.uniform(self.snr_low, self.snr_high)
        clean_amplitude = compute_amplitude(audio)
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude
        
        babbled_audio = audio * (1 - noise_amplitude_factor)
        
        audio_len = len(audio)
        babble_waveform = np.zeros(audio_len, dtype=audio.dtype)
        for i in range(self.speaker_count):
            waveform_idx = (1 + i) % self.speaker_count
            aligned_bw = align_waveform(
                babble_waveforms[waveform_idx], 
                audio_len, self.pad_noise, None)
            babble_waveform += aligned_bw
        
        babble_amplitude = compute_amplitude(babble_waveform)
        babble_waveform *= new_noise_amplitude / (babble_amplitude + 1e-14)
        babbled_audio += babble_waveform
        
        return babbled_audio

if __name__ == '__main__':
    import sys
    import soundfile as sf
    from pathlib import Path
    
    if len(sys.argv) < 3:
        print("Usage: python add_babble.py <output_dir> <babble_csv> [audio_file]")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    babble_csv = sys.argv[2]
    audio_file = sys.argv[3] if len(sys.argv) > 3 else "./tmp/test.flac"
    
    if not Path(babble_csv).exists():
        raise ValueError(f"babble CSV file not found: {babble_csv}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = "onebit/configs/test.yaml"
    config_manager = ConfigManager(config_path)
    cli_args_dict = {
        "data.aug.add_babble.probability": 1.0,
        "data.aug.add_babble.babble_csv": babble_csv
    }
    config_manager.merge_with_cli(cli_args_dict)
    
    add_babble_augmentor = AddBabbleAugmentation(config_manager)
    
    from onebit.data import audio_util
    audio_array, sr = audio_util.get_audio(audio_file, trim_sil=False)
    augmented = add_babble_augmentor(audio_array)
    
    output_path = output_dir / "add_babble.wav"
    sf.write(str(output_path), augmented, sr)
    print(f"Augmented audio saved to {output_path}")

