# encoding: utf-8
# author: Yixuan
#
#
import torch
import torch.nn as nn

from onebit.config import ConfigManager
from onebit.data import AudioBatch
from onebit.model.datatypes import BackendOutput
from onebit.model.frontend import FrontendModel
from onebit.model.backend import BackendFactory

class Model(nn.Module):
    """
    Main model class: combine frontend and backend
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager = config_manager
        self.frontend = FrontendModel(config_manager)
        self.backend = BackendFactory.create(config_manager)

        frontend_cfg = config_manager.get_model_config().frontend
        self.freeze_frontend = getattr(frontend_cfg, 'freeze_frontend', True)

    def forward(self, batch: AudioBatch) -> BackendOutput:
        if self.freeze_frontend: 
            with torch.no_grad():
                frontend_output = self.frontend(batch)
        else:
            frontend_output = self.frontend(batch)
        backend_output = self.backend(batch, frontend_output)
        return backend_output
    

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

    model = Model(config_manager)
    model.eval()

    with torch.no_grad():
        model_out = model(audio_batch)
    
    print("logits shape", model_out.logits.shape)