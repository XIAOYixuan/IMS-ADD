# TODO: need to be refactored, the design is too heavy, 
# we don't need the manager
# we only need a list of hooks directly initialized 
# in the frontend model

import torch
from typing import Dict, List, Any, Optional
from .hook_registry import HookRegistry
from .base_hook import BaseHook


class HookManager:
    
    def __init__(self, model: torch.nn.Module, hook_config: Dict[str, Any]):
        self.model = model
        self.hook_config = hook_config
        self.active_hooks: Dict[str, BaseHook] = {}
        self.enabled = hook_config.get('enabled', False)
        
        if self.enabled:
            self._setup_hooks()
    
    def _setup_hooks(self) -> None:
        hook_names = self.hook_config.get('names', [])
        
        for hook_name in hook_names:
            try:
                hook = HookRegistry.create(hook_name, hook_name)
                hook.register_hooks(self.model)
                self.active_hooks[hook_name] = hook
                print(f"Successfully registered hook: {hook_name}")
            except Exception as e:
                raise RuntimeError(f"Failed to register hook '{hook_name}': {str(e)}")
    
    def clear_batch_activations(self) -> None:
        if not self.enabled:
            return
            
        for hook in self.active_hooks.values():
            hook.clear_activations()
    
    def get_batch_activations(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        if not self.enabled:
            return None
            
        batch_activations = {}
        for hook_name, hook in self.active_hooks.items():
            activations = hook.get_activations()
            if activations:
                batch_activations[hook_name] = activations
                
        return batch_activations if batch_activations else None
    
    def save_example_to_disk(self, filepath: str = "temp.pt") -> None:
        if not self.enabled:
            print("Hooks are disabled")
            return
            
        all_activations = self.get_batch_activations()
        if all_activations:
            torch.save(all_activations, filepath)
            print(f"Saved all hook activations to {filepath}")
        else:
            print("No activations to save")
    
    def remove_all_hooks(self) -> None:
        for hook in self.active_hooks.values():
            hook.remove_hooks()
        self.active_hooks.clear()
    
    def __del__(self):
        self.remove_all_hooks()
