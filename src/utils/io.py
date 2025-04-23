import os
import json
import yaml
import pickle
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import torch
import numpy as np
from datetime import datetime
import cv2


def read_yaml_config(config_path: str) -> Any:
    """
    Read YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed configuration object
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert to namespace for dot notation access
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __repr__(self):
            keys = sorted(self.__dict__)
            items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
            return "{}({})".format(type(self).__name__, ", ".join(items))

    def dict_to_namespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = dict_to_namespace(value)
            return Namespace(**d)
        return d

    return dict_to_namespace(config)


def read_json_config(config_path: str) -> Dict[str, Any]:
    """
    Read JSON configuration file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save dictionary as JSON.

    Args:
        data: Dictionary to save
        path: Path to save JSON file
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_pickle(path: str) -> Any:
    """
    Load pickle file.

    Args:
        path: Path to pickle file

    Returns:
        Unpickled object
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data: Any, path: str) -> None:
    """
    Save object as pickle.

    Args:
        data: Object to save
        path: Path to save pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def save_model(model: torch.nn.Module, path: str) -> None:
    """
    Save PyTorch model.

    Args:
        model: PyTorch model to save
        path: Path to save model
    """
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Load weights into PyTorch model.

    Args:
        model: PyTorch model instance
        path: Path to model weights file
        device: Device to load model to

    Returns:
        Model with loaded weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict = torch.load(path, map_location=device)

    # Handle DataParallel saved models
    if list(state_dict.keys())[0].startswith('module.'):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    return model


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object of the directory
    """
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: str, extension: Optional[str] = None) -> List[str]:
    """
    List all files in a directory, optionally filtered by extension.

    Args:
        directory: Directory path
        extension: Optional file extension to filter by

    Returns:
        List of file paths
    """
    files = []
    for file in os.listdir(directory):
        if extension is None or file.endswith(extension):
            files.append(os.path.join(directory, file))
    return files


def create_run_dir(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """
    Create a uniquely named directory for a training run.

    Args:
        base_dir: Base directory
        experiment_name: Name of experiment

    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    ensure_dir(run_dir)
    return run_dir


def save_image(img: Union[torch.Tensor, np.ndarray], path: Union[str, Path], normalize: bool = True) -> None:
    """
    Save an image tensor or array to disk.

    Args:
        img: Image to save (torch.Tensor or numpy.ndarray)
        path: Path to save the image
        normalize: Whether to normalize the image from [-1, 1] to [0, 255]
    """
    # Convert torch tensor to numpy array if needed
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

        # Remove batch dimension if present
        if img.ndim == 4:
            img = img[0]

        # Convert CHW to HWC format
        if img.shape[0] in [1, 3]:  # If first dimension is channels
            img = np.transpose(img, (1, 2, 0))

    # Handle grayscale images
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        if img.ndim == 3:
            img = img.squeeze(2)  # Remove channel dimension
    elif img.ndim == 3 and img.shape[2] == 3:
        # RGB image, no change needed
        pass
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # Normalize from [-1, 1] to [0, 255] if requested
    if normalize:
        if img.min() < 0:  # Assume it's in [-1, 1] range
            img = (img + 1) * 127.5
        elif img.max() <= 1.0:  # Assume it's in [0, 1] range
            img = img * 255.0

    # Ensure image is in uint8 format
    img = img.astype(np.uint8)

    # Save using OpenCV (BGR format) or PIL (RGB format)
    if img.ndim == 2 or img.shape[2] == 1:
        # Grayscale image
        cv2.imwrite(str(path), img)
    else:
        # RGB image, convert to BGR for OpenCV
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_image_grid(imgs: List[torch.Tensor], path: Union[str, Path], nrow: int = 8, padding: int = 2) -> None:
    """
    Save a grid of images.

    Args:
        imgs: List of image tensors
        path: Path to save the grid
        nrow: Number of images per row
        padding: Padding between images
    """
    from torchvision.utils import make_grid

    # Make grid and convert to numpy
    grid = make_grid(imgs, nrow=nrow, padding=padding)
    grid = grid.permute(1, 2, 0).cpu().numpy()

    # Scale from [-1, 1] to [0, 255] if needed
    if grid.min() < 0:
        grid = (grid + 1) * 127.5
    elif grid.max() <= 1.0:
        grid = grid * 255.0

    # Save the grid
    cv2.imwrite(str(path), cv2.cvtColor(grid.astype(np.uint8), cv2.COLOR_RGB2BGR))


def tensor_to_pil(img: torch.Tensor) -> Any:
    """
    Convert a tensor to a PIL Image.

    Args:
        img: Image tensor (C,H,W)

    Returns:
        PIL Image
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    # Ensure img is on CPU and detach from graph
    img = img.cpu().detach()

    # Remove batch dimension if present
    if img.dim() == 4:
        img = img[0]

    # Scale from [-1, 1] to [0, 1] if needed
    if img.min() < 0:
        img = (img + 1) / 2

    # Convert to PIL Image
    return TF.to_pil_image(img)