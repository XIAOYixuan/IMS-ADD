# encoding: utf-8
# author: Yixuan
#
#
from onebit.config import ConfigManager
from onebit.model.frontend.base import BaseFrontendModel
from onebit.model.frontend.registry import FrontendRegistry

class FrontendFactory:

    @staticmethod
    def create(config_manager: ConfigManager) -> BaseFrontendModel:
        frontend_config = config_manager.get_model_config().get('frontend')
        frontend_name: str = frontend_config.get('name')
        # Hard coded: for hf_frontend, there's always a path
        if '/' in frontend_name:
            frontend_name = 'hf_frontend'

        FrontendRegistry._ensure_initialized()

        if FrontendRegistry.has_frontend(frontend_name):
            frontend_class = FrontendRegistry.get(frontend_name)
            return frontend_class(config_manager)
        else:
            raise ValueError(f"Unknown frontend: {frontend_name}. Available frontends: {FrontendRegistry.list_frontends()}")
