# encoding: utf-8
# author: Yixuan
#
#
from dataclasses import asdict

import torch
import torch.nn as nn
from transformers import AutoModel

from onebit.config import ConfigManager
from onebit.data.audiocollator import AudioBatch
from onebit.model.datatypes import FrontendOutput
from onebit.util import get_logger
from .hook_manager import HookManager 

logger = get_logger(__name__)
class FrontendModel(nn.Module):

    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        frontend_config = config_manager.get_model_config().frontend
        logger.info(f'Loading frontend model: {frontend_config.name}')
        self.model = AutoModel.from_pretrained(frontend_config.name)
        self.frontend_cfg = frontend_config 
        logger.info(f'front end config \n {frontend_config}')
        if getattr(self.frontend_cfg, 'freeze_frontend', True):
            for param in self.model.parameters():
                param.requires_grad = False

        hook_config = getattr(self.frontend_cfg, 'hooks', {'enabled': False})
        self.hook_manager = HookManager(self.model, hook_config)

    def forward(self, batch: AudioBatch) -> FrontendOutput:
        self.hook_manager.clear_batch_activations()
        
        out = self.model(
            input_values=batch.input_values,
            attention_mask=batch.attention_mask,
            output_hidden_states=self.frontend_cfg.output_hidden_states
        )

        last_hidden_state: torch.Tensor = out.last_hidden_state
        T = last_hidden_state.size(1)
        attention_mask = self.model._get_feature_vector_attention_mask(
            T,
            batch.attention_mask,
            add_adapter=False
        )

        return FrontendOutput(
            foutput=out,
            attention_mask=attention_mask
        )

    def get_hook_activations(self):
        """
        Useful for debug and intermedia results modification
        """
        return self.hook_manager.get_batch_activations()
    
    def save_hook_activations_example(self, filepath: str = "temp.pt"):
        self.hook_manager.save_example_to_disk(filepath)

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
    
    # Test hook functionality
    hook_activations = wavlm_frontend.get_hook_activations()
    if hook_activations:
        print("\nHook activations captured:")
        for hook_name, activations in hook_activations.items():
            print(f"  {hook_name}:")
            for layer_key, tensor in activations.items():
                print(f"    {layer_key}: {tensor.shape}")
        
        # Save example to disk
        wavlm_frontend.save_hook_activations_example("test_activations.pt")
    else:
        print("\nNo hook activations captured (hooks may be disabled)")
 