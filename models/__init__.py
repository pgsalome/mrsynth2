from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import os

from .cyclegan import CycleGANModel
from .pix2pix import Pix2PixModel
from .diffusion import DiffusionModel, LatentDiffusionModel
from .vae import VAEModel
from .model_registry import ModelRegistry

# Register model classes with the registry
ModelRegistry.register('cyclegan', CycleGANModel)
ModelRegistry.register('pix2pix', Pix2PixModel)
ModelRegistry.register('diffusion', DiffusionModel)
ModelRegistry.register('latent_diffusion', LatentDiffusionModel)
ModelRegistry.register('vae', VAEModel)


def create_model(config: Dict[str, Any], device: Optional[torch.device] = None) -> nn.Module:
    """
    Create and initialize model based on configuration.

    Args:
        config: Configuration dictionary
        device: Device to place model on

    Returns:
        Initialized model
    """
    # Get model type from config
    model_type = config['model']['name'].lower()

    # Determine device if not provided
    if device is None:
        gpu_ids = config.get('gpu_ids', '')
        if isinstance(gpu_ids, str):
            gpu_ids = [int(id) for id in gpu_ids.split(',') if id.strip()]

        if torch.cuda.is_available() and gpu_ids:
            device = torch.device(f'cuda:{gpu_ids[0]}')
        else:
            device = torch.device('cpu')

    # Create model using registry
    model = ModelRegistry.create(model_type, config)

    # Move model to device
    model.to(device)

    # Initialize schedulers if needed
    if hasattr(model, 'setup_schedulers'):
        model.setup_schedulers()

    return model


def load_model(model_type: str, checkpoint_path: str, config: Optional[Dict[str, Any]] = None,
               device: Optional[torch.device] = None) -> nn.Module:
    """
    Load model from checkpoint.

    Args:
        model_type: Type of model to load
        checkpoint_path: Path to checkpoint file
        config: Configuration for model (optional if included in checkpoint)
        device: Device to place model on

    Returns:
        Loaded model
    """
    # If config is not provided, try to load from checkpoint directory
    if config is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, 'config.json')

        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"No config provided and no config file found at {config_path}")

    # Create model
    model = create_model(config, device)

    # Load weights
    if hasattr(model, 'load_networks'):
        # For models with custom load methods
        model.load_networks(checkpoint_path)
    else:
        # For standard PyTorch models
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    # Set to evaluation mode
    model.eval()

    return model