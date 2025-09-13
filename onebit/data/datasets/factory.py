# encoding: utf-8
# author: Yixuan
#
#

from onebit.data.datasets.base import BaseDataset
from onebit.config import ConfigManager
from onebit.data.datasets.registry import DatasetRegistry

class DatasetFactory:

    @staticmethod
    def create(split: str, config_manager: ConfigManager) -> BaseDataset:
        dataset_cfg = config_manager.get_data_config().get('dataset')
        dataset_name: str = getattr(dataset_cfg, 'type', 'audio')
        
        DatasetRegistry._ensure_initialized()
        
        if DatasetRegistry.has_dataset(dataset_name):
            dataset_cls = DatasetRegistry.get(dataset_name)
            return dataset_cls(split, config_manager)
        else:
            raise ValueError(f"Unknown backend: {dataset_name}. Available backends: {DatasetRegistry.list_datasets()}")