# encoding: utf-8
# author: Yixuan
#
#
from dataclasses import asdict

import torch
import torch.nn as nn
from transformers import AutoModel

from onebit.config import ConfigManager
from onebit.data import AudioBatch
from onebit.util import get_logger
from onebit.model.datatypes import FrontendOutput
from onebit.model.frontend.base import BaseFrontendModel
from onebit.model.frontend.registry import FrontendRegistry

logger = get_logger(__name__)

@FrontendRegistry.register('hf_frontend')
class FrontendModel(BaseFrontendModel):

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        frontend_config = config_manager.get_model_config().frontend
        logger.info(f'Loading frontend model: {frontend_config.name}')
        self.model = AutoModel.from_pretrained(frontend_config.name)
        self.frontend_cfg = frontend_config 
        logger.info(f'front end config \n {frontend_config}')
        self.freeze_frontend = getattr(self.frontend_cfg, 'freeze_frontend', True)
        if self.freeze_frontend:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, audio_batch: AudioBatch) -> FrontendOutput:

        if self.freeze_frontend:
            self.model.eval()
            with torch.no_grad():
                out = self.model(
                    input_values=audio_batch.input_values,
                    attention_mask=audio_batch.attention_mask,
                    output_hidden_states=self.frontend_cfg.output_hidden_states
                )
        else:
            out = self.model(
                input_values=audio_batch.input_values,
                attention_mask=audio_batch.attention_mask,
                output_hidden_states=self.frontend_cfg.output_hidden_states
            )

        last_hidden_state: torch.Tensor = out.last_hidden_state
        T = last_hidden_state.size(1)
        attention_mask = self.model._get_feature_vector_attention_mask(
            T,
            audio_batch.attention_mask,
            add_adapter=False
        )

        return FrontendOutput(
            foutput=out,
            attention_mask=attention_mask
        )

if __name__ == '__main__':
    B, T = 3, 64000
    input_values = torch.randn((B, T), dtype=torch.float32)
    attention_mask = torch.ones((B, T), dtype=torch.bool)
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
    #device = torch.device('cpu') 
    config_path = "onebit/configs/test.yaml"
    config_manager = ConfigManager(config_path)
    wavlm_frontend = FrontendModel(config_manager)

    wavlm_frontend.to(device)
    audio_batch = audio_batch.to(device)

    wavlm_frontend.eval()
    with torch.no_grad():
        frontend_out: FrontendOutput = wavlm_frontend(audio_batch)

    print("last_hidden_state shape:", frontend_out.foutput.last_hidden_state.shape) # type: ignore
    print("number of layers:", len(frontend_out.foutput.hidden_states)) # type: ignore
    print("shape of layer 0 hidden_state:", frontend_out.foutput.hidden_states[0].shape) # type: ignore
    print("feat attention mask shape:", frontend_out.attention_mask.shape)
    