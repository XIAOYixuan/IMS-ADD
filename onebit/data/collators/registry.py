
# encoding: utf-8
# author: Yixuan
#
#

import importlib
import os
import pkgutil
from typing import Dict, Type, Callable

from onebit.data.collators.base import BaseCollator
from onebit.util import get_logger
logger = get_logger(__name__)

class CollatorRegistry:

    _collators: Dict[str, Type[BaseCollator]] = {}
    _initialized = False

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseCollator]], Type[BaseCollator]]:

        def decorator(collator_cls: Type[BaseCollator]) -> Type[BaseCollator]:
            cls._collators[name] = collator_cls 
            return collator_cls 
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseCollator]:
        cls._ensure_initialized()
        if name not in cls._collators:
            raise ValueError(f"collator class {name} not found in registry")
        return cls._collators[name]
    
    @classmethod
    def list_collators(cls) -> list[str]:
        cls._ensure_initialized()
        return list(cls._collators.keys())

    @classmethod
    def has_collator(cls, name: str) -> bool:
        cls._ensure_initialized()
        return name in cls._collators
    
    @classmethod
    def _ensure_initialized(cls):
        if not cls._initialized:
            cls._auto_discover_collators()
            cls._initialized = True
    
    @classmethod
    def _auto_discover_collators(cls):
        import onebit.data.collators as collator_package
        collator_dir = os.path.dirname(collator_package.__file__)
        
        for _, module_name, is_pkg in pkgutil.iter_modules([collator_dir]):
            if module_name in ['__init__', 'factory', 'registry', 'base']:
                continue
            
            try:
                logger.debug(f"loaded module {module_name}")
                importlib.import_module(f'onebit.data.collators.{module_name}')
            except ImportError as e:
                logger.debug(f"failed to import collator module {module_name}: {e}")