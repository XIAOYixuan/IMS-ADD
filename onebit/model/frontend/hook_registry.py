import importlib
import os
import pkgutil
from typing import Dict, Type, Callable

from .base_hook import BaseHook
from onebit.util import get_logger

logger = get_logger(__name__)


class HookRegistry:

    _hooks: Dict[str, Type[BaseHook]] = {}
    _initialized = False

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseHook]], Type[BaseHook]]:
        def decorator(hook_cls: Type[BaseHook]) -> Type[BaseHook]:
            cls._hooks[name] = hook_cls
            return hook_cls
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseHook]:
        cls._ensure_initialized()
        if name not in cls._hooks:
            raise ValueError(f"Hook class {name} not found in registry")
        return cls._hooks[name]
    
    @classmethod
    def create(cls, name: str, *args, **kwargs) -> BaseHook:
        hook_cls = cls.get(name)
        return hook_cls(*args, **kwargs)
    
    @classmethod
    def list_hooks(cls) -> list[str]:
        cls._ensure_initialized()
        return list(cls._hooks.keys())

    @classmethod
    def has_hook(cls, name: str) -> bool:
        cls._ensure_initialized()
        return name in cls._hooks
    
    @classmethod
    def _ensure_initialized(cls):
        if not cls._initialized:
            cls._auto_discover_hooks()
            cls._initialized = True
    
    @classmethod
    def _auto_discover_hooks(cls):
        import onebit.model.frontend as frontend_package
        frontend_dir = os.path.dirname(frontend_package.__file__)
        
        for _, module_name, is_pkg in pkgutil.iter_modules([frontend_dir]):
            if module_name in ['__init__', 'frontend', 'base_hook', 'hook_registry', 'hook_manager']:
                continue
            
            try:
                logger.debug(f"Loading hook module {module_name}")
                importlib.import_module(f'onebit.model.frontend.{module_name}')
            except ImportError as e:
                logger.debug(f"Failed to import hook module {module_name}: {e}")
