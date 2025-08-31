# encoding: utf-8
# author: Yixuan
#
#

from pathlib import Path
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from typing import Dict, List, Any
import pandas as pd
from dataclasses import dataclass

from onebit.config import ConfigManager
from onebit.data.datasets.base import BaseDataset
from onebit.data.datasets.registry import DatasetRegistry 
from onebit.data.augmentors.augmentor import Augmentor
from onebit.data.postprocessors.postprocessor import PostProcessor
from onebit.data import audio_util
from onebit.util import get_logger
logger = get_logger(__name__) 

@dataclass
class AudioSample:
    """Data class representing an audio sample with its metadata."""
    uttid: str
    audio_path: str
    origin_ds: str
    speaker: str
    attacker: str
    label: str


@dataclass
class AudioSampleWithTensors(AudioSample):
    """Data class representing an audio sample with its metadata and tensors."""
    audio_array: np.ndarray # used in AutoFeatureExtractor 
    label_tensor: torch.Tensor

@DatasetRegistry.register("audio")
class AudioDataset(BaseDataset):

    def __init__(self,
                 split: str,
                 config_manager: ConfigManager):
        """
        Args:
            split: Dataset split ('train', 'dev', 'test')
            config: the config for all
        """
        # Initialize parent class with common functionality
        super().__init__(split, config_manager)
        self.data_conf = self.config_manager.get_data_config()
        
        self.sample_rate = self.config_manager.get_sample_rate()
        base_path = self.config_manager.get_dataset_path()
        self.dataset_dir = Path(base_path)

        if hasattr(self.data_conf, 'max_samples'):
            self.max_samples = int(self.data_conf.dataset.max_samples)
        else:
            self.max_samples = int(self.data_conf.dataset.max_length * self.sample_rate)
        
        self.metadata: List[AudioSample] = self._load_metadata()

        if len(self.metadata) == 0:
            raise ValueError(f"No data found for split {split} in {self.dataset_dir}")
        
        self.post_processor = PostProcessor.from_config(self.config_manager)
        self.augmentor = Augmentor.from_config(self.config_manager)
    
    def _encode_label(self, label: str) -> int:
        if label.lower() == 'bonafide':
            return 1
        elif label.lower() == 'spoof':
            return 0
        else:
            raise ValueError(f'Invalid label: [{label}], must be [bonafide, spoof]')
    
    def _load_metadata(self) -> List[AudioSample]:
        """
        Load metadata from .tsv and .txt files.
        Returns:
            List of AudioSample objects.
        """
        logger.info(f"Loading from {self.dataset_dir}")
        tsv_file = self.dataset_dir / f"{self.split}.tsv"
        txt_file = self.dataset_dir / f"{self.split}.txt"
        
        if not tsv_file.exists():
            raise FileNotFoundError(f"{tsv_file} not found")
        if not txt_file.exists():
            raise FileNotFoundError(f"{txt_file} not found")

        # TODO: have max length and min length
        # Filter out too long or too short audios
        tsv_data: pd.DataFrame = pd.read_csv(tsv_file, sep='\t', header=None, names=['uttid', 'audio_path'])
        txt_data: pd.DataFrame = pd.read_csv(txt_file, sep='\t', header=None, 
                        names=['uttid', 'origin_ds', 'speaker', 'attacker', 'label'])
        
        if not self.validate_dataset(tsv_data, txt_data):
            raise ValueError(f"Dataset validation failed for split {self.split}")

        merged_data = pd.merge(tsv_data, txt_data, on='uttid', how='inner')

        metadata = []
        for _, row in merged_data.iterrows():
            audio_sample = AudioSample(
                uttid=str(row['uttid']),
                audio_path=str(row['audio_path']),
                origin_ds=str(row['origin_ds']),
                speaker=str(row['speaker']),
                attacker=str(row['attacker']),
                label=str(row['label'])
            )
            metadata.append(audio_sample)

        return metadata

    def _load_audio(self, audio_path: str):
        audio, sr = audio_util.get_audio(audio_path, 
                                to_mono = True,
                                norm=True,
                                trim_sil=self.data_conf.dataset.trim_sil)
        # audio shape [audio_length]
        duration = len(audio) / sr
        if duration < self.data_conf.dataset.min_length:
            logger.info(f"Audio too short {duration} Path: {audio_path}")
            return None

        return audio
    
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        if self.post_processor is not None:
            audio = self.post_processor(audio)
        
        if self.split == 'train' and self.augmentor is not None:
            audio = self.augmentor(audio)

        return audio

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> AudioSampleWithTensors:

        while True:
            sample = self.metadata[idx]
            audio = self._load_audio(sample.audio_path)
            if audio is not None:
                break
            logger.info(f"Failed to load audio, resampling...")
        
        audio = self._process_audio(audio)
        label = self._encode_label(sample.label)
        #audio_tensor = torch.from_numpy(audio).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return AudioSampleWithTensors(
            uttid=sample.uttid,
            audio_path=sample.audio_path,
            origin_ds=sample.origin_ds,
            speaker=sample.speaker,
            attacker=sample.attacker,
            label=sample.label,
            audio_array=audio,
            label_tensor=label_tensor
        )

    def validate_dataset(self, tsv_data: pd.DataFrame, txt_data: pd.DataFrame) -> bool:
        """
        This function would check whether 
        1. in txt file, all id in the txt or tsv are unique, no duplication
        2. for each uttid in the txt file, there's a corresponding one in the tsv 
        3. all audio paths are unique
        4. same amount of samples in txt/tsv
        """
        # check 1: unique ids 
        txt_uttids = txt_data['uttid'].tolist()
        if len(txt_uttids) != len(set(txt_uttids)):
            duplicates = [uttid for uttid in set(txt_uttids) if txt_uttids.count(uttid) > 1]
            logger.error(f"Duplicate uttids found in txt file: {duplicates}")
            return False
        
        tsv_uttids = tsv_data['uttid'].tolist()
        if len(tsv_uttids) != len(set(tsv_uttids)):
            duplicates = [uttid for uttid in set(tsv_uttids) if tsv_uttids.count(uttid) > 1]
            logger.error(f"Duplicate uttids found in tsv file: {duplicates}")
            return False
        
        # check 2: match txt and tsv
        txt_uttid_set = set(txt_uttids)
        tsv_uttid_set = set(tsv_uttids)
        
        missing_in_tsv = txt_uttid_set - tsv_uttid_set
        if missing_in_tsv:
            logger.error(f"Uttids in txt file missing from tsv file: {list(missing_in_tsv)[:10]}...")
            return False
        missing_in_txt = tsv_uttid_set - txt_uttid_set
        if missing_in_txt:
            logger.error(f"Uttids in tsv file missing from txt file: {list(missing_in_txt)[:10]}...")
            return False
        
        # check 3: unique audio paths
        audio_paths = tsv_data['audio_path'].tolist()
        if len(audio_paths) != len(set(audio_paths)):
            duplicates = [path for path in set(audio_paths) if audio_paths.count(path) > 1]
            logger.error(f"Duplicate audio paths found: {duplicates[:10]}...")
            return False
        
        # Check 4: same amount of samples in txt/tsv
        if len(txt_data) != len(tsv_data):
            logger.error(f"Sample count mismatch: txt {len(txt_data)}, tsv {len(tsv_data)}")
            return False
        
        logger.info(f"Dataset validation passed for split '{self.split}': {len(txt_data)} samples")
        return True

    def print_stat(self) -> None: # for module test, generated by claude-4-sonnet(cursor)
        """Print comprehensive statistics about the dataset."""
        print(f"\n{'='*60}")
        print(f"Dataset Statistics for Split: {self.split.upper()}")
        print(f"{'='*60}")
        
        # Basic dataset info
        total_samples = len(self.metadata)
        print(f"Total Samples: {total_samples:,}")
        print(f"Dataset Directory: {self.dataset_dir}")
        print(f"Sample Rate: {self.sample_rate:,} Hz")
        
        # Label distribution
        print(f"\n{'-'*40}")
        print("Label Distribution:")
        print(f"{'-'*40}")
        label_counts = {}
        for sample in self.metadata:
            label = sample.label.lower()
            label_counts[label] = label_counts.get(label, 0) + 1
        
        for label, count in sorted(label_counts.items()):
            percentage = (count / total_samples) * 100
            print(f"{label.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        # Origin dataset distribution
        print(f"\n{'-'*40}")
        print("Origin Dataset Distribution:")
        print(f"{'-'*40}")
        origin_counts = {}
        for sample in self.metadata:
            origin = sample.origin_ds
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
        
        for origin, count in sorted(origin_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_samples) * 100
            print(f"{origin}: {count:,} ({percentage:.1f}%)")
        
        # Speaker statistics
        print(f"\n{'-'*40}")
        print("Speaker Statistics:")
        print(f"{'-'*40}")
        unique_speakers = set(sample.speaker for sample in self.metadata)
        print(f"Unique Speakers: {len(unique_speakers):,}")
        
        # Speaker distribution (top 10)
        speaker_counts = {}
        for sample in self.metadata:
            speaker = sample.speaker
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 Speakers by Sample Count:")
        for speaker, count in top_speakers:
            percentage = (count / total_samples) * 100
            print(f"  {speaker}: {count:,} ({percentage:.1f}%)")
        
        # Attacker statistics (for spoof samples)
        print(f"\n{'-'*40}")
        print("Attacker Statistics:")
        print(f"{'-'*40}")
        spoof_samples = [s for s in self.metadata if s.label.lower() == 'spoof']
        if spoof_samples:
            attacker_counts = {}
            for sample in spoof_samples:
                attacker = sample.attacker
                attacker_counts[attacker] = attacker_counts.get(attacker, 0) + 1
            
            print(f"Total Spoof Samples: {len(spoof_samples):,}")
            print(f"Unique Attackers: {len(set(s.attacker for s in spoof_samples)):,}")
            
            for attacker, count in sorted(attacker_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(spoof_samples)) * 100
                print(f"  {attacker}: {count:,} ({percentage:.1f}%)")
        else:
            print("No spoof samples found.")
        
        # Audio processing configuration
        print(f"\n{'-'*40}")
        print("Processing Configuration:")
        print(f"{'-'*40}")
        print(f"Post-processing Enabled: {self.config_manager.is_post_processing_enabled()}")
        print(f"Augmentation Enabled: {self.config_manager.is_augmentation_enabled() and self.split == 'train'}")
        if self.post_processor is not None:
            print(f"PostProcessor: {len(self.post_processor.postprocessors)} postprocessors loaded")
        if self.augmentor is not None:
            print(f"Augmentor: {len(self.augmentor.augmentors)} augmentors loaded")
        print(f"Trim Silence: {self.data_conf.dataset.trim_sil}")
        
        # File path validation (quick check without loading audio)
        print(f"\n{'-'*40}")
        print("File Path Validation:")
        print(f"{'-'*40}")
        
        existing_files = 0
        sample_check_size = min(100, len(self.metadata))
        
        for i in range(sample_check_size):
            sample = self.metadata[i]
            if Path(sample.audio_path).exists():
                existing_files += 1
        
        existence_rate = (existing_files / sample_check_size) * 100
        print(f"Files Checked: {sample_check_size}")
        print(f"Files Found: {existing_files} ({existence_rate:.1f}%)")
        
        if existence_rate < 100:
            print(f"⚠️  Warning: {sample_check_size - existing_files} files not found in sample")
        else:
            print("✅ All sampled files exist")
        
        print(f"\n{'='*60}")
        print("End of Dataset Statistics")
        print(f"{'='*60}\n")

if __name__ == '__main__':
    import argparse
    # module test: generated by claude-4-sonnet(cursor)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test AudioDataset and print statistics')
    parser.add_argument('--config', type=str, default='onebit/configs/test.yaml',
                        help='Path to config file (default: onebit/configs/test.yaml)')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Override dataset name in config (e.g., "ASVspoof2019")')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'dev', 'test'],
                        help='Dataset split to load (default: train)')
    
    args = parser.parse_args()
    
    try:
        # Load config manager
        print(f"Loading config from: {args.config}")
        config_manager = ConfigManager(args.config)
        
        # Override dataset name if provided
        if args.dataset_name and args.dataset_name.startswith('dataset_meta/'):
            args.dataset_name = args.dataset_name[len('dataset_meta/'):]
        if args.dataset_name:
            print(f"Overriding dataset name to: {args.dataset_name}")
            # Access the config and modify the dataset name
            original_name = config_manager.config.data.dataset.name
            config_manager.config.data.dataset.name = args.dataset_name
            print(f"Dataset name changed from '{original_name}' to '{args.dataset_name}'")
        
        # Create dataset instance
        print(f"Creating AudioDataset for split: {args.split}")
        dataset = AudioDataset(split=args.split, config_manager=config_manager)

        # Print statistics
        dataset.print_stat()
        
        print(f"✅ Module test completed successfully!")
        print(f"Dataset contains {len(dataset)} samples")
        
    except Exception as e:
        print(f"❌ Error during module test: {e}")
        import traceback
        traceback.print_exc()