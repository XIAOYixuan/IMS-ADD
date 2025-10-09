# encoding: utf-8
# author: claude-4-sonnet(cursor) generated based on ../backend/base.py

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseHook(ABC):
    """Abstract base class for all hooks."""
    
    def __init__(self, name: str):
        self.name = name
        self.hooks = []  # Store hook handles for cleanup
        self.activations = {}  # Store captured activations
        
    @abstractmethod
    def register_hooks(self, model: torch.nn.Module) -> None:
        """Register hooks on the specified model."""
        pass
    
    @abstractmethod  
    def get_target_modules(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Find and return target modules to hook."""
        pass
    
    def clear_activations(self) -> None:
        """Clear stored activations (auto-called after each batch)."""
        self.activations.clear()
        
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get current batch activations."""
        return self.activations.copy()
    
    def save_example_to_disk(self, filepath: str = "temp.pt") -> None:
        """Example function to save activations to disk."""
        if self.activations:
            # Move to CPU before saving
            cpu_activations = {}
            for key, tensor in self.activations.items():
                if isinstance(tensor, torch.Tensor):
                    cpu_activations[key] = tensor.cpu()
                else:
                    cpu_activations[key] = tensor
            
            torch.save(cpu_activations, filepath)
            print(f"Saved activations to {filepath}")
        else:
            print("No activations to save")
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
