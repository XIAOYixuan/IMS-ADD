# encoding: utf-8
# author: Yixuan
#
#

import importlib
import os
import pkgutil
from typing import Dict, Type, Callable

from onebit.loss.base import BaseLoss 
from onebit.util import get_logger
logger = get_logger(__name__)

class LossRegistry:

    _losses: Dict[str, Type[BaseLoss]] = {}
    _initialized = False

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseLoss]], Type[BaseLoss]]:

        def decorator(loss_cls: Type[BaseLoss]) -> Type[BaseLoss]:
            cls._losses[name] = loss_cls 
            return loss_cls 
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseLoss]:
        cls._ensure_initialized()
        if name not in cls._losses:
            raise ValueError(f"loss class {name} not found in registry")
        return cls._losses[name]
    
    @classmethod
    def list_losses(cls) -> list[str]:
        cls._ensure_initialized()
        return list(cls._losses.keys())

    @classmethod
    def has_loss(cls, name: str) -> bool:
        cls._ensure_initialized()
        return name in cls._losses
    
    @classmethod
    def _ensure_initialized(cls):
        if not cls._initialized:
            cls._auto_discover_backends()
            cls._initialized = True
    
    @classmethod
    def _auto_discover_backends(cls):
        import onebit.loss as loss_pkg
        loss_dir = os.path.dirname(loss_pkg.__file__)
        logger.debug(f'checking {loss_dir}')
        for _, module_name, is_pkg in pkgutil.iter_modules([loss_dir]):
            if module_name in ['__init__', 'base', 'factory', 'registry', 'datatypes']:
                continue
            
            try:
                logger.debug(f"loaded module onebit.loss.{module_name}")
                importlib.import_module(f'onebit.loss.{module_name}')
            except ImportError as e:
                logger.debug(f"failed to import loss module {module_name}: {e}")