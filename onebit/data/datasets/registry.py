
# encoding: utf-8
# author: Yixuan
#
#

import importlib
import os
import pkgutil
from typing import Dict, Type, Callable

from onebit.util import get_logger
from onebit.data.datasets.base import BaseDataset
logger = get_logger(__name__)

class DatasetRegistry:

    _datasets: Dict[str, Type[BaseDataset]] = {}
    _initialized = False

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseDataset]], Type[BaseDataset]]:

        def decorator(dataset_cls: Type[BaseDataset]) -> Type[BaseDataset]:
            cls._datasets[name] = dataset_cls 
            return dataset_cls 
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseDataset]:
        cls._ensure_initialized()
        if name not in cls._datasets:
            raise ValueError(f"model class {name} not found in registry")
        return cls._datasets[name]
    
    @classmethod
    def list_datasets(cls) -> list[str]:
        cls._ensure_initialized()
        return list(cls._datasets.keys())

    @classmethod
    def has_dataset(cls, name: str) -> bool:
        cls._ensure_initialized()
        return name in cls._datasets
    
    @classmethod
    def _ensure_initialized(cls):
        if not cls._initialized:
            cls._auto_discover_datasets()
            cls._initialized = True
    
    @classmethod
    def _auto_discover_datasets(cls):
        import onebit.data.datasets as dataset_package
        dataset_dir = os.path.dirname(dataset_package.__file__)
        
        for _, module_name, is_pkg in pkgutil.iter_modules([dataset_dir]):
            if module_name in ['__init__', 'factory', 'registry']:
                continue
            
            try:
                logger.debug(f"loaded module {module_name}")
                importlib.import_module(f'onebit.data.datasets.{module_name}')
            except ImportError as e:
                logger.debug(f"failed to import dataset module {module_name}: {e}")