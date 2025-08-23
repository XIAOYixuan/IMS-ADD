# encoding: utf-8
# author: Yixuan
#
#

import importlib
import os
import pkgutil
from typing import Dict, Type, Callable

from onebit.model.backend.base import BaseBackendModel
from onebit.util import get_logger
logger = get_logger(__name__)

class BackendRegistry:

    _backends: Dict[str, Type[BaseBackendModel]] = {}
    _initialized = False

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseBackendModel]], Type[BaseBackendModel]]:

        def decorator(bakend_cls: Type[BaseBackendModel]) -> Type[BaseBackendModel]:
            cls._backends[name] = bakend_cls
            return bakend_cls
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseBackendModel]:
        cls._ensure_initialized()
        if name not in cls._backends:
            raise ValueError(f"model class {name} not found in registry")
        return cls._backends[name]
    
    @classmethod
    def list_backend(cls) -> list[str]:
        cls._ensure_initialized()
        return list(cls._backends.keys())

    @classmethod
    def has_backend(cls, name: str) -> bool:
        cls._ensure_initialized()
        return name in cls._backends
    
    @classmethod
    def _ensure_initialized(cls):
        if not cls._initialized:
            cls._auto_discover_backends()
            cls._initialized = True
    
    @classmethod
    def _auto_discover_backends(cls):
        import onebit.model.backend as backend_package
        backend_dir = os.path.dirname(backend_package.__file__)
        
        for _, module_name, is_pkg in pkgutil.iter_modules([backend_dir]):
            if module_name in ['__init__', 'base', 'factory', 'registry']:
                continue
            
            try:
                logger.debug(f"loaded module {module_name}")
                importlib.import_module(f'onebit.model.backend.{module_name}')
            except ImportError as e:
                logger.debug(f"failed to import backend module {module_name}: {e}")