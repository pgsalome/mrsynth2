import os
import json
import yaml
import pickle
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists.

    Args:
        path: Directory path

    Returns:
        Path object of created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save data as JSON.

    Args:
        data: Data to save
        path: File path
    """
    path = Path(path)

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert NumPy types to Python types
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (dict)):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        return obj

    # Save JSON
    with open(path, 'w') as f:
        json.dump(convert_numpy(data), f, indent=2)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON.

    Args:
        path: File path

    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save data as YAML.

    Args:
        data: Data to save
        path: File path
    """
    path = Path(path)

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save YAML
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from YAML.

    Args:
        path: File path

    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_pickle(data: Any, path: Union[str, Path]) -> None:
    """
    Save data as pickle.

    Args:
        data: Data to save
        path: File path
    """
    path = Path(path)

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Load data from pickle.

    Args:
        path: File path

    Returns:
        Loaded data
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_image(tensor: torch.Tensor, path: Union[str, Path], normalize: bool = True) -> None:
    """
    Save a tensor as an image.

    Args:
        tensor: Image tensor (C,H,W) or (B,C,H,W)
        path: File path
        normalize: Whether to normalize to [0,255]
    """
    path = Path(path)

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure tensor is on CPU and detached from graph
    tensor = tensor.detach().cpu()

    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Convert to NumPy array
    if tensor.dim() == 3:
        # (C,H,W) to (H,W,C)
        array = tensor.permute(1, 2, 0).numpy()
    else:
        array = tensor.numpy()

    # Normalize to [0,1]
    if normalize:
        if array.min() < 0 or array.max() > 1:
            array = (array - array.min()) / (array.max() - array.min() + 1e-8)

    # Scale to [0,255] and convert to uint8
    array = (array * 255).astype(np.uint8)

    # Save image
    Image.fromarray(array).save(path)


def load_image(path: Union[str, Path], grayscale: bool = False) -> torch.Tensor:
    """
    Load an image as a tensor.

    Args:
        path: File path
        grayscale: Whether to load as grayscale

    Returns:
        Image tensor (C,H,W)
    """
    # Load image
    if grayscale:
        img = Image.open(path).convert('L')
        # Add channel dimension
        tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
    else:
        img = Image.open(path).convert('RGB')
        # Convert to tensor and normalize
        tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0

    return tensor


def save_image_grid(tensors: List[torch.Tensor], path: Union[str, Path],
                    nrow: int = 8, padding: int = 2, normalize: bool = True) -> None:
    """
    Save a grid of images.

    Args:
        tensors: List of image tensors (C,H,W)
        path: File path
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize to [0,255]
    """
    try:
        from torchvision.utils import make_grid
    except ImportError:
        raise ImportError("torchvision is required for save_image_grid")

    # Create grid
    grid = make_grid(tensors, nrow=nrow, padding=padding, normalize=normalize)

    # Save grid
    save_image(grid, path, normalize=False)


def save_model(model: torch.nn.Module, path: Union[str, Path], optimizer: Optional[torch.optim.Optimizer] = None,
               epoch: Optional[int] = None, loss: Optional[float] = None,
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a model.

    Args:
        model: Model to save
        path: File path
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        metadata: Additional metadata
    """
    path = Path(path)

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if loss is not None:
        checkpoint['loss'] = loss

    if metadata is not None:
        checkpoint['metadata'] = metadata

    # Save checkpoint
    torch.save(checkpoint, path)


def load_model(model: torch.nn.Module, path: Union[str, Path],
               optimizer: Optional[torch.optim.Optimizer] = None,
               device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load a model.

    Args:
        model: Model to load into
        path: File path
        optimizer: Optimizer to load into
        device: Device to load model onto

    Returns:
        Checkpoint dictionary
    """
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint