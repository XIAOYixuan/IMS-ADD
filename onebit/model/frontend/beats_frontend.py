# encoding: utf-8
# author: Yixuan
#
#
from dataclasses import asdict

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput 
from transformers import AutoModel

from onebit.config import ConfigManager
from onebit.data import AudioBatch
from onebit.util import get_logger
from onebit.model.datatypes import FrontendOutput
from onebit.model.frontend.hooks.hook_manager import HookManager
from onebit.model.frontend.base import BaseFrontendModel
from onebit.model.frontend.registry import FrontendRegistry

logger = get_logger(__name__)

@FrontendRegistry.register('beats')
class BEATsFrontend(BaseFrontendModel):

    def __init__(self, config_manager):
        from onebit.model.frontend.beats.BEATs import BEATsModel
        super().__init__(config_manager)
        frontend_cfg = config_manager.get_model_config().frontend
        model_path = frontend_cfg.get('model_path', None)
        if model_path is None:
            raise ValueError(f"model_path not found")
        self.freeze_frontend = frontend_cfg.get('freeze_frontend', True)
        self.model = BEATsModel(model_path)

    def forward(self, audio_batch: AudioBatch) -> FrontendOutput:

        if self.freeze_frontend:
            self.model.eval()
            with torch.no_grad():
                feats, mask, layer_results = self.model(audio_batch.input_values, 
                                         audio_batch.attention_mask)
        else: 
            feats, mask, layer_results = self.model(audio_batch.input_values, 
                               audio_batch.attention_mask)

        foutput = BaseModelOutput(last_hidden_state=feats, hidden_states=layer_results)
        return FrontendOutput(foutput=foutput, attention_mask=mask)
    
if __name__ == '__main__':
    import sys
    B, T = 3, 64000
    input_values = torch.randn((B, T), dtype=torch.float32)
    attention_mask = None 
    label_tensors = torch.zeros((B,), dtype=torch.long)
    
    audio_batch = AudioBatch(
        input_values=input_values,
        attention_mask=attention_mask,
        label_tensors=label_tensors,
        labels = [''],
        uttids=[''],
        speakers=[''],
        attackers=[''],
        origin_ds=[''],
        audio_paths=['']
    )

    device = torch.device('cuda')
    print(sys.argv)
    config_path = sys.argv[-1]
    config_manager = ConfigManager(config_path)
    beats = BEATsFrontend(config_manager)
    beats.to(device)
    audio_batch = audio_batch.to(device)

    foutput: FrontendOutput = beats(audio_batch) 
    print(foutput.attention_mask)
    for key in foutput.foutput:
        #N, T, D = 3, 48, 768
        if isinstance(foutput.foutput[key], torch.Tensor):
            print(key, foutput.foutput[key].shape)
            if torch.isnan(foutput.foutput[key]).any():
                raise ValueError(f'NaN exists')
        elif isinstance(foutput.foutput[key], tuple):
            items = foutput.foutput[key]
            print(len(items))
            for i, item in enumerate(items):
                print(i, item.shape)
        else:
            print(type(foutput.foutput[key]))