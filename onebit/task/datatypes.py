# encoding: utf-8
# author: Yixuan
#
#

from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class BaseStatus:
    debug: Any = None

@dataclass
class TrainStatus(BaseStatus):
    # might be useful for debug
    # it can carry more info
    loss: float = 0.

@dataclass
class ValStatus(TrainStatus):
    eer: float = float('inf')
    thresh: float = 0.

@dataclass
class ModelCheckpoint:
    epoch: int
    batch_count: int
    best_eer: float
    #optimizer_state_dict: Dict[str, Any]
    frontend_frozen: bool
    model_state_dict: Optional[Dict[str, Any]] = None  # full model state when frontend not frozen
    backend_state_dict: Optional[Dict[str, Any]] = None  # backend only when frontend frozen