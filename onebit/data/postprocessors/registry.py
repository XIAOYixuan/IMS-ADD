
# encoding: utf-8
# author: Yixuan
#
#

import importlib
import os
import pkgutil
from typing import Dict, Type, Callable

from onebit.data.postprocessors.base import BasePostProcessor
from onebit.util import get_logger
logger = get_logger(__name__)

class PostProcessorRegistry:

    _postprocessors: Dict[str, Type[BasePostProcessor]] = {}
    _initialized = False

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BasePostProcessor]], Type[BasePostProcessor]]:

        def decorator(postprocessor_cls: Type[BasePostProcessor]) -> Type[BasePostProcessor]:
            cls._postprocessors[name] = postprocessor_cls 
            return postprocessor_cls 
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BasePostProcessor]:
        cls._ensure_initialized()
        if name not in cls._postprocessors:
            raise ValueError(f"postprocessor class {name} not found in registry")
        return cls._postprocessors[name]
    
    @classmethod
    def list_postprocessors(cls) -> list[str]:
        cls._ensure_initialized()
        return list(cls._postprocessors.keys())

    @classmethod
    def has_postprocessor(cls, name: str) -> bool:
        cls._ensure_initialized()
        return name in cls._postprocessors
    
    @classmethod
    def _ensure_initialized(cls):
        if not cls._initialized:
            cls._auto_discover_postprocessors()
            cls._initialized = True
    
    @classmethod
    def _auto_discover_postprocessors(cls):
        import onebit.data.postprocessors as postprocessor_package
        postprocessor_dir = os.path.dirname(postprocessor_package.__file__)
        
        for _, module_name, is_pkg in pkgutil.iter_modules([postprocessor_dir]):
            if module_name in ['__init__', 'factory', 'registry', 'base', 'postprocessor']:
                continue
            
            try:
                logger.debug(f"loaded module {module_name}")
                importlib.import_module(f'onebit.data.postprocessors.{module_name}')
            except ImportError as e:
                logger.debug(f"failed to import postprocessor module {module_name}: {e}")