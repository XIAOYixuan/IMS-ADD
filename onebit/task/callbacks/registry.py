# encoding: utf-8
# author: Yixuan
#
#

import importlib
import os
import pkgutil
from typing import Dict, Type, Callable

from onebit.task.callbacks.base import BaseCallback
from onebit.util import get_logger
logger = get_logger(__name__)

class CallbackRegistry:

    _callbacks: Dict[str, Type[BaseCallback]] = {}
    _initialized = False

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseCallback]], Type[BaseCallback]]:

        def decorator(callback_cls: Type[BaseCallback]) -> Type[BaseCallback]:
            cls._callbacks[name] = callback_cls
            return callback_cls
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseCallback]:
        cls._ensure_initialized()
        if name not in cls._callbacks:
            raise ValueError(f"callback class {name} not found in registry")
        return cls._callbacks[name]
    
    @classmethod
    def list_callbacks(cls) -> list[str]:
        cls._ensure_initialized()
        return list(cls._callbacks.keys())

    @classmethod
    def has_callback(cls, name: str) -> bool:
        cls._ensure_initialized()
        return name in cls._callbacks
    
    @classmethod
    def _ensure_initialized(cls):
        if not cls._initialized:
            cls._auto_discover_callbacks()
            cls._initialized = True
    
    @classmethod
    def _auto_discover_callbacks(cls):
        import onebit.task.callbacks as callbacks_pkg
        callbacks_dir = os.path.dirname(callbacks_pkg.__file__)
        logger.debug(f'checking {callbacks_dir}')
        
        for _, module_name, is_pkg in pkgutil.iter_modules([callbacks_dir]):
            if module_name in ['__init__', 'base', 'factory', 'registry']:
                continue
            
            try:
                logger.debug(f"loaded module onebit.task.callbacks.{module_name}")
                importlib.import_module(f'onebit.task.callbacks.{module_name}')
            except ImportError as e:
                logger.debug(f"failed to import callback module {module_name}: {e}")
