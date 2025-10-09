# encoding: utf-8
# author: Yixuan
#
#

import importlib
import os
import pkgutil
from typing import Dict, Type, Callable

from onebit.model.frontend.base import BaseFrontendModel
from onebit.util import get_logger
logger = get_logger(__name__)

class FrontendRegistry:

    _frontends: Dict[str, Type[BaseFrontendModel]] = {}
    _initialized = False

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseFrontendModel]], Type[BaseFrontendModel]]:
        def decorator(frontend_cls: Type[BaseFrontendModel]) -> Type[BaseFrontendModel]:
            cls._frontends[name] = frontend_cls
            return frontend_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseFrontendModel]:
        cls._ensure_initialized()
        if name not in cls._frontends:
            raise ValueError(f"frontend model class {name} not found in registry")
        return cls._frontends[name]

    @classmethod
    def list_frontends(cls) -> list[str]:
        cls._ensure_initialized()
        return list(cls._frontends.keys())

    @classmethod
    def has_frontend(cls, name: str) -> bool:
        cls._ensure_initialized()
        return name in cls._frontends

    @classmethod
    def _ensure_initialized(cls):
        if not cls._initialized:
            cls._auto_discover_frontends()
            cls._initialized = True

    @classmethod
    def _auto_discover_frontends(cls):
        import onebit.model.frontend as frontend_package
        frontend_dir = os.path.dirname(frontend_package.__file__)

        for _, module_name, is_pkg in pkgutil.iter_modules([frontend_dir]):
            if module_name in ['__init__', 
                               'base', 
                               'factory', 
                               'registry',
                               'hooks']:
                continue
            try:
                logger.debug(f"loaded frontend module {module_name}")
                importlib.import_module(f'onebit.model.frontend.{module_name}')
            except ImportError as e:
                logger.debug(f"failed to import frontend module {module_name}: {e}")
