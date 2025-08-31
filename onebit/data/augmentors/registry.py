# encoding: utf-8
# author: Yixuan
#
#

import importlib
import os
import pkgutil
from typing import Dict, Type, Callable

from onebit.data.augmentors.base import BaseAugmentor
from onebit.util import get_logger
logger = get_logger(__name__)

class AugmentorRegistry:

    _augmentors: Dict[str, Type[BaseAugmentor]] = {}
    _initialized = False

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseAugmentor]], Type[BaseAugmentor]]:

        def decorator(augmentor_cls: Type[BaseAugmentor]) -> Type[BaseAugmentor]:
            cls._augmentors[name] = augmentor_cls 
            return augmentor_cls 
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseAugmentor]:
        cls._ensure_initialized()
        if name not in cls._augmentors:
            raise ValueError(f"model class {name} not found in registry")
        return cls._augmentors[name]
    
    @classmethod
    def list_augmentors(cls) -> list[str]:
        cls._ensure_initialized()
        return list(cls._augmentors.keys())

    @classmethod
    def has_augmentor(cls, name: str) -> bool:
        cls._ensure_initialized()
        return name in cls._augmentors
    
    @classmethod
    def _ensure_initialized(cls):
        if not cls._initialized:
            cls._auto_discover_augmentors()
            cls._initialized = True
    
    @classmethod
    def _auto_discover_augmentors(cls):
        import onebit.data.augmentors as augmentor_package
        augmentor_dir = os.path.dirname(augmentor_package.__file__)
        
        for _, module_name, is_pkg in pkgutil.iter_modules([augmentor_dir]):
            if module_name in ['__init__', 'factory', 'registry', 'base', 'augmentor']:
                continue
            
            try:
                logger.debug(f"loaded module {module_name}")
                importlib.import_module(f'onebit.data.augmentors.{module_name}')
            except ImportError as e:
                logger.debug(f"failed to import augmentor module {module_name}: {e}")