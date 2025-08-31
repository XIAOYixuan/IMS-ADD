# Import all augmentor modules to ensure they are registered
from . import rawboost

# Import main classes for external use
from .augmentor import Augmentor
from .factory import AugmentorFactory
from .base import BaseAugmentor
from .registry import AugmentorRegistry

__all__ = [
    'Augmentor',
    'AugmentorFactory', 
    'BaseAugmentor',
    'AugmentorRegistry'
]
