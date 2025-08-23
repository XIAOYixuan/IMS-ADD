# Adpated from: https://github.com/piotrkawa/deepfake-whisper-features
import torch
import numpy as np
import librosa
import soundfile as sf
import logging

logging.basicConfig(level=logging.WARNING)

# Set the logging level for Numba
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

SAMPLING_RATE = 16_000

def trim_silence(audio: np.ndarray, 
                 threshold: float = 30, 
                 win_length: int = 1024, 
                 shift_length: int = 256) -> np.ndarray:
    """
    Trims the leading and trailing silence of the given audio.

    :param audio: A numpy.ndarray representing the audio signal.
    :param threshold: The threshold in decibels used for silence detection. Default is 30 dB.
    :param win_length: The window length for analysis in points. Default is 1024.
    :param shift_length: The shift length for the analysis window in points. Default is 256.
    :return: A numpy.ndarray of the trimmed audio.
    """
    # Use librosa's trim function to remove leading and trailing silence based on the specified parameters.
    trimmed_audio, _ = librosa.effects.trim(
        y=audio,
        top_db=threshold,
        frame_length=win_length,
        hop_length=shift_length
    )
    
    return trimmed_audio

def get_audio(
    audio_path,
    to_mono=True,
    norm=True,
    trim_sil=True,
    target_rms=-20,
):
    """
    Load an audio file using soundfile, optionally resample with librosa, 
    optionally convert to mono, and optionally RMS-normalize to target_rms dB.
    
    :param audio_path: Path to the audio file.
    :param to_mono: If True, keep only the first channel (as in original code).
    :param norm: If True, apply RMS-based normalization to target_rms dB.
    :param trim_sil: If True, trim leading and trailing silence.
    :param target_rms: Target RMS level in dB (default -20 dB).
    :param frame_offset: Number of frames to skip from the beginning.
    :param num_frames: How many frames to read. None or negative => read all.
    :return: A tuple (waveform, sample_rate), 
             where waveform is a NumPy array of shape (num_samples,) or (channels, num_samples).
    """
    #with sf.SoundFile(audio_path, 'r') as f:
    #    if frame_offset > 0:
    #        f.seek(frame_offset)
    #    data = f.read(frames=num_frames, dtype='float32', always_2d=False)
    #    sample_rate = f.samplerate
    data, sr = librosa.load(audio_path, sr=SAMPLING_RATE, mono=to_mono)

    if trim_sil:
        data = trim_silence(data)

    if norm:
        rms = np.sqrt(np.mean(np.square(data)))
        target_rms_linear = 10 ** (target_rms / 20.0)
        scaling_factor = target_rms_linear / rms
        data = data * scaling_factor
        data = np.clip(data, -1.0, 1.0)
    
    #data = data.reshape(1, -1)
    #data_tensor = torch.from_numpy(data)
    #return data_tensor, sample_rate
    return data, sr

if __name__ == "__main__":
    # NOTE: need to set random seed for testing 
    # read 4 seconds
    audio_path = '/mount/arbeitsdaten54/projekte/deepfake/fad/data/release_in_the_wild/21945.wav'
    #audio = read_audio(audio_path, 4)
    audio, sample_rate = get_audio(audio_path, to_mono=True, trim_sil=True)
    print(audio.shape) #[length]
