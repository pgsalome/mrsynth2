import torch
from typing import Dict, Any, Optional

from models.cycle_gan import CycleGANModel
from models.pix2pix import Pix2PixModel
from models.diffusion import DiffusionModel, LatentDiffusionModel
from models.vae import VAEModel


def get_model(config: Dict[str, Any], device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Create and initialize model based on configuration.

    Args:
        config: Configuration dictionary
        device: Device to place model on (if None, will be determined automatically)

    Returns:
        Initialized model
    """
    model_name = config["model"]["name"].lower()

    # Create model based on type
    if model_name == "cyclegan":
        model = CycleGANModel(config)
    elif model_name == "pix2pix.json":
        model = Pix2PixModel(config)
    elif model_name == "diffusion.json":
        model = DiffusionModel(config)
    elif model_name == "latent_diffusion.json":
        model = LatentDiffusionModel(config)
    elif model_name == "vae":
        model = VAEModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Determine device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model to device
    model.to(device)

    # Initialize model (e.g., setup optimizers, schedulers)
    if hasattr(model, 'setup_schedulers'):
        model.setup_schedulers()

    return model


def load_model_from_checkpoint(
        model_dir: str,
        epoch: str = "best",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load a model from checkpoint.

    Args:
        model_dir: Directory containing checkpoint files
        epoch: Epoch to load (e.g., "best", "latest", or specific number)
        config: Configuration dictionary (if None, will attempt to load from model_dir)
        device: Device to place model on (if None, will be determined automatically)

    Returns:
        Loaded model
    """
    from pathlib import Path
    from utils.io import read_json_config

    # Determine device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration if not provided
    if config is None:
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")

        config = read_json_config(str(config_path))

    # Create model
    model = get_model(config, device)

    # Load checkpoint
    if hasattr(model, "load_networks"):
        model.load_networks(epoch)
    else:
        # Standard PyTorch model loading
        checkpoint_path = Path(model_dir) / f"{epoch}_model.pth"
        if not checkpoint_path.exists():
            # Try alternatives
            if epoch == "best":
                checkpoint_path = Path(model_dir) / "best_model.pth"
            elif epoch == "latest":
                checkpoint_path = Path(model_dir) / "latest_model.pth"
            elif epoch == "final":
                checkpoint_path = Path(model_dir) / "final_model.pth"

        if checkpoint_path.exists():
            state_dict = torch.load(str(checkpoint_path), map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: No checkpoint found for epoch '{epoch}'. Using initialized model.")

    # Set to evaluation mode
    model.eval()

    return model