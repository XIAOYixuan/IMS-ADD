# encoding: utf-8
# author: Yixuan
#
#
from onebit.config import ConfigManager
from onebit.loss.base import BaseLoss 
from onebit.loss.registry import LossRegistry

class LossFactory:

    @staticmethod
    def create(config_manager: ConfigManager) -> BaseLoss:
        loss_config = config_manager.get_loss_config()
        loss_name: str = loss_config.name
        
        LossRegistry._ensure_initialized()
        
        if LossRegistry.has_loss(loss_name):
            backend_class = LossRegistry.get(loss_name)
            return backend_class(config_manager)
        else:
            raise ValueError(f"Unknown loss: {loss_name}. Available losses: {LossRegistry.list_losses()}")

if __name__ == '__main__':
    import torch
    from onebit.config import ConfigManager
    from onebit.loss.registry import LossRegistry
    from onebit.data.audiocollator import AudioBatch
    from onebit.model.datatypes import FrontendOutput, BackendOutput
    
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

    t, d = 199, 768
    from transformers.utils.generic import ModelOutput
    hf_model_out = ModelOutput(
        last_hidden_state=torch.randn(B, t, d),
        hidden_states=tuple(torch.randn(B, t, d) for _ in range(13)) 
    )

    frontend_out = FrontendOutput(
        foutput=hf_model_out,
        attention_mask=torch.ones((B, t))
    )

    backend_out = BackendOutput(
        logits=torch.randn(B),
        predictions=torch.randn(B),
        frontend_output=frontend_out
    )

    device = torch.device('cuda')
    #device = torch.device('cpu') 
    config_path = "onebit/configs/test.yaml"
    config_manager = ConfigManager(config_path)

    loss: BaseLoss = LossFactory.create(config_manager)
    loss.eval()

    with torch.no_grad():
        loss_out = loss(audio_batch, frontend_out, backend_out)

    print(type(loss_out)) 
    print(loss_out)