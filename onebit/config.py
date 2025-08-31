from omegaconf import OmegaConf, DictConfig, ListConfig
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import os

"""
# Command line usage with shortened names
python train.py --config config.yaml \
  --exp.name=my_experiment \
  --data.dataset.name=cd_add_ted \
  --data.loader.bz=64 \
  --data.aug.enabled=true \
  --model.frontend.name=microsoft/wavlm-base \
  --model.backend.name=pool_feature \
  --loss.name=bcewithlogits \
  --train.hyperparameters.lr=0.001 \
  --train.hyperparameters.epochs=100
"""

class ConfigManager:
    """
    Configuration manager for OneBit audio deepfake detection system.

    This class handles loading, validating, and managing YAML configuration files
    for all aspects of the experiment including data, model, and training settings.
    Uses OmegaConf for enhanced functionality including structured access,
    CLI integration, and configuration inheritance.

    The configuration system is designed to be lightweight and user-friendly:
    - Users only need to set essential parameters
    - Sensible defaults are provided for all technical settings
    - Advanced options are hidden but can be accessed if needed
    - Uses shortened parameter names for CLI convenience
    """

    def __init__(self, config_path: str):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: Union[DictConfig, ListConfig] = self._load_config()
        # Note: Validation happens after loading to allow inheritance to resolve
        self._validate_config()

    def _load_config(self) -> Union[DictConfig, ListConfig]:
        """
        Load configuration from YAML file using OmegaConf.

        Returns:
            OmegaConf configuration object
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        # Load the config file
        config = OmegaConf.load(self.config_path)

        # Handle defaults inheritance if present
        if hasattr(config, 'defaults') and config.defaults:
            # Create a new config with defaults resolved
            resolved_config = OmegaConf.create()

            # Load and merge each default config
            for default_name in config.defaults:
                if default_name == '_self_':
                    # Merge the current config
                    resolved_config = OmegaConf.merge(resolved_config, config)
                else:
                    # Load the default config file
                    default_path = self.config_path.parent / f"{default_name}.yaml"
                    if default_path.exists():
                        default_config = OmegaConf.load(default_path)
                        resolved_config = OmegaConf.merge(resolved_config, default_config)
                    else:
                        print(f"Warning: Default config file not found: {default_path}")

            return resolved_config

        return config

    def _validate_config(self):
        """Validate configuration structure and values."""
        required_sections = ['exp', 'data', 'model', 'loss']

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")

        # Validate data section
        if not hasattr(self.config.data, 'dataset'):
            raise ValueError("Missing 'dataset' section in data configuration")

        # Validate dataset configuration
        dataset_config = self.config.data.dataset
        if not hasattr(dataset_config, 'name'):
            raise ValueError("Missing required dataset field: name")

        if not hasattr(dataset_config, 'root_dir'):
            raise ValueError("Missing required dataset field: root_dir")

    def get_data_config(self) -> DictConfig:
        return self.config.data

    def get_model_config(self) -> DictConfig:
        return self.config.model

    def get_loss_config(self) -> DictConfig:
        return self.config.loss

    def get_exp_config(self) -> DictConfig:
        return self.config.exp
    
    def merge_with_cli(self, cli_args):
        """
        Merge configuration with command line arguments using OmegaConf.

        Args:
            cli_args: Either a dictionary of arguments or a list of strings
                     (like command-line arguments)
        """
        if isinstance(cli_args, dict):
            # Convert dictionary to dotlist format
            dotlist = [f"{key}={value}" for key, value in cli_args.items()]
            cli_config = OmegaConf.from_dotlist(dotlist)
        elif isinstance(cli_args, (list, tuple)):
            cli_config = OmegaConf.from_cli(list(cli_args))
        else:
            raise ValueError("cli_args must be a dictionary or list/tuple of strings")

        self.config = OmegaConf.merge(self.config, cli_config)

    def save_config(self, save_path: Path):
        """
        Save current configuration to file using OmegaConf.

        Args:
            save_path: Path to save the configuration
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(self.config, save_path)

    def print_config(self):
        """Pretty print configuration using OmegaConf."""
        print("=" * 50)
        print("OneBit Configuration")
        print("=" * 50)
        print(OmegaConf.to_yaml(self.config))

    def get_dataset_path(self) -> str:
        """
        Get the full dataset path.

        Returns:
            Full path to the dataset directory
        """
        dataset_config = self.config.data.dataset
        root_dir = dataset_config.root_dir
        dataset_name = dataset_config.name

        return str(Path(root_dir) / dataset_name)

    def get_dataset_name(self) -> str:
        """
        Get the dataset name.

        Returns:
            Dataset name
        """
        return self.config.data.dataset.name

    def get_dataset_splits(self) -> List[str]:
        """
        Get the dataset's available splits.

        Returns:
            Dataset split (train, dev, test)
        """
        dataset_path = Path(self.get_dataset_path())
        avail_splits = []
        for split in ['train', 'dev', 'test']:
            txt_path = dataset_path / f"{split}.txt"
            if not txt_path.exists(): continue
            tsv_path = dataset_path / f"{split}.tsv"
            if not tsv_path.exists(): continue
            avail_splits.append(split)
        return avail_splits

    def get_train_split_path(self) -> str:
        return str(Path(self.get_dataset_path()) / "train")

    def get_val_split_path(self) -> str:
        return str(Path(self.get_dataset_path()) / "dev")

    def get_test_split_path(self) -> str:
        return str(Path(self.get_dataset_path()) / "test")

    def get_sample_rate(self) -> int:
        dataset_config = self.config.data.dataset
        return getattr(dataset_config, 'sample_rate', 16000)

    def get_batch_size(self, split: str = 'train') -> int:
        loader_config = self.config.data.loader

        if split == 'train':
            return getattr(loader_config, 'bz', 128)
        elif split == 'dev':
            return getattr(loader_config, 'val_bz', 64)
        elif split == 'test':
            return getattr(loader_config, 'test_bz', 192)
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_augmentation_config(self) -> DictConfig:
        return self.config.data.aug

    def get_post_processing_config(self) -> DictConfig:
        return self.config.data.post

    def get_experiment_name(self) -> str:
        return self.config.exp.name

    def get_experiment_seed(self) -> int:
        return getattr(self.config.exp, 'seed', 42)

    def get_log_dir(self) -> str:
        return getattr(self.config.exp, 'log_dir', './logs')

    def get_output_dir(self) -> str:
        return getattr(self.config.exp, 'output_dir', './outputs')

    def get_checkpoint_dir(self) -> str:
        return getattr(self.config.exp, 'checkpoint_dir', './checkpoints')

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self.config[key]  # type: ignore

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self.config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self.config.get(key, default) # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert OmegaConf configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        result = OmegaConf.to_container(self.config, resolve=True)
        if isinstance(result, dict):
            return result  # type: ignore
        else:
            raise ValueError("Configuration is not a dictionary")

    def print_user_config(self):
        """
        Print a user-friendly summary of the configuration.
        Shows only the essential settings that users typically care about.
        """
        print("=" * 60)
        print("OneBit Configuration Summary")
        print("=" * 60)

        # Experiment settings
        print(f"Experiment: {self.get_experiment_name()}")
        print(f"Random Seed: {self.get_experiment_seed()}")
        print(f"Log Directory: {self.get_log_dir()}")
        print(f"Output Directory: {self.get_output_dir()}")
        print(f"Checkpoint Directory: {self.get_checkpoint_dir()}")

        print("\nData Settings:")
        print(f"  Dataset Name: {self.get_dataset_name()}")
        print(f"  Dataset Path: {self.get_dataset_path()}")
        print(f"  Dataset Split: {self.get_dataset_splits()}")
        print(f"  Sample Rate: {self.get_sample_rate()} Hz")
        print(f"  Batch Sizes:")
        print(f"    Train: {self.get_batch_size('train')}")
        print(f"    Validation: {self.get_batch_size('dev')}")
        print(f"    Test: {self.get_batch_size('test')}")
        print(f"  Augmentation: {'Enabled' if self.is_augmentation_enabled() else 'Disabled'}")
        print(f"  Post-processing: {'Enabled' if self.is_post_processing_enabled() else 'Disabled'}")

        # Model settings
        print("\nModel Settings:")
        model_config = self.get_model_config()
        if hasattr(model_config, 'frontend'):
            frontend = model_config.frontend  # type: ignore
            print(f"  Frontend Name: {getattr(frontend, 'name', 'N/A')}")
            print(f"  Freeze Frontend: {getattr(frontend, 'freeze_frontend', 'N/A')}")
            print(f"  Output Hidden States: {getattr(frontend, 'output_hidden_states', 'N/A')}")
        if hasattr(model_config, 'backend'):
            backend = model_config.backend  # type: ignore
            print(f"  Backend Name: {getattr(backend, 'name', 'N/A')}")
        
        # Loss settings
        print("\nLoss Settings:")
        loss_config = self.get_loss_config()
        print(f"  Loss Function: {getattr(loss_config, 'name', 'N/A')}")
        if hasattr(loss_config, 'real_weight'):
            print(f"  Real Weight: {getattr(loss_config, 'real_weight', 'N/A')}")

    def print_cli_reference(self):
        """
        Print CLI parameter reference for convenience.
        """
        print("=" * 60)
        print("OneBit CLI Parameters Reference")
        print("=" * 60)
        print("Use these parameters for CLI convenience:")
        print()
        print("Experiment Settings:")
        print("  exp.name=<name>              # Experiment name")
        print("  exp.seed=<seed>              # Random seed")
        print("  exp.log_dir=<path>           # Log directory")
        print("  exp.output_dir=<path>        # Output directory")
        print("  exp.checkpoint_dir=<path>    # Checkpoint directory")
        print()
        print("Data Settings:")
        print("  data.dataset.name=<name>     # Dataset name (REQUIRED)")
        print("  data.dataset.split=<split>   # Dataset split (for testing: train/dev/test)")
        print("  data.loader.bz=<size>        # Training batch size")
        print("  data.loader.val_bz=<size>    # Validation batch size")
        print("  data.loader.test_bz=<size>   # Test batch size")
        print("  data.aug.enabled=<bool>      # Enable/disable augmentation")
        print("  data.aug.probability=<float> # Augmentation probability")
        print("  data.post.enabled=<bool>     # Enable/disable post-processing")
        print()
        print("Model Settings:")
        print("  model.frontend.name=<name>           # Frontend model name (e.g., microsoft/wavlm-base)")
        print("  model.frontend.freeze_frontend=<bool> # Freeze frontend weights")
        print("  model.frontend.output_hidden_states=<bool> # Output hidden states")
        print("  model.backend.name=<name>            # Backend model name")
        print()
        print("Loss Settings:")
        print("  loss.name=<name>                     # Loss function name")
        print("  loss.real_weight=<float>             # Weight for real class")
        print()
        print("Training Settings:")
        print("  train.hyperparameters.lr=<rate>     # Learning rate")
        print("  train.hyperparameters.epochs=<num>  # Number of epochs")
        print("  train.hyperparameters.patience=<num> # Early stopping patience")
        print("  train.hyperparameters.min_delta=<value> # Minimum delta")
        print("  train.hyperparameters.optimizer=<name> # Optimizer name")
        print("  train.hyperparameters.weight_decay=<value> # Weight decay")
        print()
        print("Example CLI usage:")
        print("  python train.py --config config.yaml \\")
        print("    --exp.name=my_experiment \\")
        print("    --data.dataset.name=cd_add_ted \\")
        print("    --data.loader.bz=64 \\")
        print("    --data.aug.enabled=true \\")
        print("    --model.frontend.name=microsoft/wavlm-base \\")
        print("    --model.backend.name=pool_feature \\")
        print("    --loss.name=bcewithlogits \\")
        print("    --train.hyperparameters.lr=0.001 \\")
        print("    --train.hyperparameters.epochs=100")
        print("=" * 60)


def example_test():
    """
    This function shows how to:
    1. Load a configuration file
    2. Access configuration values using structured access
    3. Merge CLI arguments
    4. Print and save configurations
    """
    try:
        # Example 1: Basic configuration loading
        print("=== Example 1: Basic Configuration Loading ===")
        config_path = "onebit/configs/test.yaml"
        config_manager = ConfigManager(config_path)

        # Print user-friendly summary
        config_manager.print_user_config()

        # Show CLI reference
        config_manager.print_cli_reference()

        # Example 2: Using helper methods
        print("\n=== Example 2: Using Helper Methods ===")
        print(f"Dataset path: {config_manager.get_dataset_path()}")
        print(f"Sample rate: {config_manager.get_sample_rate()}")
        print(f"Train batch size: {config_manager.get_batch_size('train')}")
        print(f"Val batch size: {config_manager.get_batch_size('dev')}")
        print(f"Test batch size: {config_manager.get_batch_size('test')}")
        print(f"Augmentation enabled: {config_manager.is_augmentation_enabled()}")
        print(f"Post-processing enabled: {config_manager.is_post_processing_enabled()}")
        print(f"Experiment name: {config_manager.get_experiment_name()}")
        print(f"Log directory: {config_manager.get_log_dir()}")

        # Example 3: CLI argument merging
        print("\n=== Example 3: CLI Argument Merging ===")

        # Method 1: Using dictionary (programmatic)
        cli_args_dict = {
            "data.dataset.name": "new_dataset",
            "data.loader.bz": 32,
            "exp.name": "modified_experiment",
            "model.frontend.name": "microsoft/wavlm-large",
            "loss.name": "crossentropy"
        }

        print("Before CLI merge:")
        print(f"  Dataset path: {config_manager.get_dataset_path()}")
        print(f"  Train batch size: {config_manager.get_batch_size('train')}")
        print(f"  Experiment name: {config_manager.get_experiment_name()}")

        config_manager.merge_with_cli(cli_args_dict)

        print("After CLI merge (dictionary):")
        print(f"  Dataset path: {config_manager.get_dataset_path()}")
        print(f"  Train batch size: {config_manager.get_batch_size('train')}")
        print(f"  Experiment name: {config_manager.get_experiment_name()}")

        # Method 2: Using list (like command-line arguments)
        cli_args_list = [
            "data.dataset.name=another_dataset",
            "data.loader.bz=16",
            "exp.name=cli_list_experiment",
            "model.backend.name=transformer",
            "loss.real_weight=5.0"
        ]

        config_manager.merge_with_cli(cli_args_list)

        print("After CLI merge (list):")
        print(f"  Dataset path: {config_manager.get_dataset_path()}")
        print(f"  Train batch size: {config_manager.get_batch_size('train')}")
        print(f"  Experiment name: {config_manager.get_experiment_name()}")

        # Example 4: Configuration sections
        print("\n=== Example 4: Configuration Sections ===")
        data_config = config_manager.get_data_config()
        model_config = config_manager.get_model_config()
        loss_config = config_manager.get_loss_config()
        experiment_config = config_manager.get_exp_config()
        aug_config = config_manager.get_augmentation_config()
        print("Augmentation config type is ", type(aug_config))

        print(f"Data config - augmentation enabled: {data_config.aug.enabled}") # type: ignore
        print(f"Model config - frontend name: {model_config.frontend.name}") # type: ignore
        print(f"Model config - backend name: {model_config.backend.name}") # type: ignore
        print(f"Loss config - name: {loss_config.name}") # type: ignore
        print(f"Experiment config - seed: {experiment_config.seed}") # type: ignore

        # Example 5: Save and print configuration
        print("\n=== Example 5: Configuration Output ===")
        print("Full configuration structure:")
        config_manager.print_config()

        # Save configuration to a temporary file
        temp_save_path = Path("temp_config_save.yaml")
        config_manager.save_config(temp_save_path)
        print(f"\nConfiguration saved to: {temp_save_path}")

        print("\n✅ All examples completed successfully!")

    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        print("Make sure the onebit/configs/test.yaml file exists.")
    except Exception as e:
        print(f"❌ Error during configuration test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the example test when the file is executed directly
    example_test()
