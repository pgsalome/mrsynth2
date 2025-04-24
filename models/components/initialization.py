"""Weight initialization utilities."""

import torch
import torch.nn as nn
from typing import Union, List, Optional, Callable


def init_weights(net: nn.Module, init_type: str = 'normal', init_gain: float = 0.02) -> None:
    """
    Initialize network weights.

    Args:
        net: Network to initialize
        init_type: Initialization method ('normal', 'xavier', 'kaiming', or 'orthogonal')
        init_gain: Gain factor for initialization
    """

    def init_func(m):
        classname = m.__class__.__name__
        # Initialize conv and linear layers
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')

            # Initialize bias if it exists
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        # Initialize batch norm layers
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    # Apply initialization
    net.apply(init_func)


def init_model(model: nn.Module, init_type: str = 'normal', init_gain: float = 0.02,
               gpu_ids: Optional[List[int]] = None) -> nn.Module:
    """
    Initialize and optimize a model.

    Args:
        model: Model to initialize
        init_type: Initialization method
        init_gain: Gain factor for initialization
        gpu_ids: List of GPU IDs to use

    Returns:
        Initialized model
    """
    # Initialize weights
    init_weights(model, init_type, init_gain)

    # Move to GPU(s) if available
    if gpu_ids and torch.cuda.is_available():
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, gpu_ids)
        model = model.cuda()

    return model


def set_requires_grad(nets: Union[nn.Module, List[nn.Module]], requires_grad: bool = False) -> None:
    """
    Set requires_grad for all parameters in networks.

    Args:
        nets: Network or list of networks
        requires_grad: Whether gradients are required
    """
    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_model(model: nn.Module, path: str, epoch: Optional[int] = None) -> None:
    """
    Save model weights.

    Args:
        model: Model to save
        path: Path to save model
        epoch: Current epoch (if None, not included in filename)
    """
    import os
    from pathlib import Path

    # Create directory if it doesn't exist
    save_dir = Path(path).parent
    os.makedirs(save_dir, exist_ok=True)

    # If model is DataParallel, save the module
    if isinstance(model, nn.DataParallel):
        model = model.module

    # Save the model
    if epoch is not None:
        path = str(Path(path).with_stem(f"{Path(path).stem}_epoch{epoch}"))

    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str, strict: bool = True) -> nn.Module:
    """
    Load model weights.

    Args:
        model: Model to load weights into
        path: Path to model weights
        strict: Whether to strictly enforce that the keys in state_dict match

    Returns:
        Model with loaded weights
    """
    # If model is DataParallel, load into the module
    if isinstance(model, nn.DataParallel):
        model = model.module

    # Load state dict
    state_dict = torch.load(path, map_location='cpu')

    # Handle DataParallel saved models
    if list(state_dict.keys())[0].startswith('module.'):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict

    # Load state dict into model
    model.load_state_dict(state_dict, strict=strict)

    return model