# encoding: utf-8
# author: Yixuan
#
#

from pathlib import Path
import numpy as np
import random
import torch
from typing import List
from dataclasses import dataclass

from onebit.config import ConfigManager
from onebit.data.datasets.registry import DatasetRegistry 
from onebit.data.datasets.audiodataset import AudioDataset, AudioSampleWithTensors
from onebit.data import audio_util
from onebit.util import get_logger

logger = get_logger(__name__)

@dataclass
class BarlowTwinSampleWithTensors(AudioSampleWithTensors):
    twin_audio_array: np.ndarray

@DatasetRegistry.register("barlow_twin")
class BarlowTwinDataset(AudioDataset):
    """
    Returns two differently augmented ver of the same input.
    It randomly samples two augmentation directories for each utterance,
    ensuring that both dirs have the audio file with the same basename.
    """
    def __init__(self,
                 split: str,
                 config_manager: ConfigManager):
        super().__init__(split, config_manager)
        
        self.augmentation_dirs = self._load_augmentation_directories()
        logger.info(f"BarlowTwinDataset initialized with "
                f"{len(self.metadata)} samples and " #type: ignore
                f"{len(self.augmentation_dirs)} augmentation directories")

    def _load_augmentation_directories(self) -> List[Path]:
        aug_dirs_list = self.config_manager.get_data_config().aug_dirs
        
        aug_dirs = [Path(self.dataset_dir) / aug_dir for aug_dir in aug_dirs_list]
        return aug_dirs        

    def _load_audio_from_augmentation_dir(self, 
            base_filename: str, 
            aug_dir: Path) -> np.ndarray:
        aug_file_path = aug_dir / base_filename
        
        if not aug_file_path.exists():
            raise ValueError(f"Path not found: {aug_file_path}")
        
        return self._load_audio(str(aug_file_path))

    def __getitem__(self, idx: int) -> BarlowTwinSampleWithTensors:
        """
        Get a Barlow Twin sample with two differently augmented versions of the same audio.
        
        Args:
            idx: Sample index
            
        Returns:
            BarlowTwinSample with two augmented audio versions and shared label
        """
        sample = self.metadata[idx]
        base_filename = Path(sample.audio_path).name
            
        aug_dirs = random.choices(self.augmentation_dirs, k=2)
        
        twin1_audio = self._load_audio_from_augmentation_dir(
            base_filename, aug_dirs[0])
        twin2_audio = self._load_audio_from_augmentation_dir(
            base_filename, aug_dirs[1])
            
        twin1_audio = self._process_audio(twin1_audio)
        twin2_audio = self._process_audio(twin2_audio)
        
        label_tensor = torch.tensor(self._encode_label(sample.label), dtype=torch.long)
        
        return BarlowTwinSampleWithTensors(
            uttid=sample.uttid,
            audio_path=sample.audio_path,
            origin_ds=sample.origin_ds,
            speaker=sample.speaker,
            attacker=sample.attacker,
            label=sample.label,
            audio_array=twin1_audio,  
            twin_audio_array=twin2_audio,
            label_tensor=label_tensor,
        )


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test BarlowTwinDataset')
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
            original_name = config_manager.config.data.dataset.name
            config_manager.config.data.dataset.name = args.dataset_name
            print(f"Dataset name changed from '{original_name}' to '{args.dataset_name}'")
        
        # Create dataset instance
        print(f"Creating BarlowTwinDataset for split: {args.split}")
        dataset = BarlowTwinDataset(split=args.split, config_manager=config_manager)
        
        # Test loading a sample
        if len(dataset) > 0:
            print("Testing sample loading...")
            sample: BarlowTwinSampleWithTensors = dataset[0]
            print(f"Sample 0 loaded successfully:")
            print(f"  UttID: {sample.uttid}")
            print(f"  Label: {sample.label}")
            print(f"  Primary audio shape: {sample.audio_array.shape}")
            print(f"  Twin audio shape: {sample.twin_audio_array.shape}")
            
            # Test that we actually get different augmentations (at least sometimes)
            sample2 = dataset[0]  # Same index, should potentially give different augmentations
            primary_equal = np.array_equal(sample.audio_array, sample2.audio_array)
            twin_equal = np.array_equal(sample.twin_audio_array, sample2.twin_audio_array)
            print(f"  Primary identical on repeated access: {primary_equal}")
            print(f"  Twin identical on repeated access: {twin_equal}")
            if not (primary_equal and twin_equal):
                print("  ✅ Different augmentations produced on repeated access")
        
        print(f"✅ Module test completed successfully!")
        print(f"Dataset contains {len(dataset)} samples")
        
    except Exception as e:
        print(f"❌ Error during module test: {e}")
        import traceback
        traceback.print_exc()
