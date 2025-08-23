# encoding: utf-8
# author: Yixuan
#
#
from onebit.config import ConfigManager
from onebit.model.backend.base import BaseBackendModel
from onebit.model.backend.registry import BackendRegistry

class BackendFactory:

    @staticmethod
    def create(config_manager: ConfigManager) -> BaseBackendModel:
        backend_config = config_manager.get_model_config().get('backend')
        backend_name: str = backend_config.name
        
        BackendRegistry._ensure_initialized()
        
        if BackendRegistry.has_backend(backend_name):
            backend_class = BackendRegistry.get(backend_name)
            return backend_class(config_manager)
        else:
            raise ValueError(f"Unknown backend: {backend_name}. Available backends: {BackendRegistry.list_backend()}")