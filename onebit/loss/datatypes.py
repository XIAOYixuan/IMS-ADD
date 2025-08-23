# encoding: utf-8
# author: Yixuan
#
#

from dataclasses import dataclass

import torch

@dataclass
class LossOutput:

    loss: torch.Tensor 

    def to(self, device: torch.device, non_blocking: bool = True):
        self.loss = self.loss.to(device, non_blocking=non_blocking)
        return self