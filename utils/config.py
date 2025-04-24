import os
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
import copy


def read_json_config(config_path: str) -> Dict[str, Any]:
    """Read JSON configuration file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary.

    Args:
        base_dict: Base dictionary to be updated
        update_dict: Dictionary with values to update

    Returns:
        Updated dictionary
    """
    result = copy.deepcopy(base_dict)

    for k, v in update_dict.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = copy.deepcopy(v)

    return result


def load_model_config(model_type: str, base_config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and merge base and model-specific configurations.

    Args:
        model_type: Type of model (e.g., 'cyclegan', 'diffusion.json', 'vae')
        base_config_path: Path to base config file (defaults to 'config/base.json')

    Returns:
        Merged configuration dictionary
    """
    # Set default base config path if not provided
    if base_config_path is None:
        base_config_path = os.path.join('config', 'base.json')

    # Load base config
    base_config = read_json_config(base_config_path)

    # Construct model config path
    model_config_path = os.path.join('config', f'{model_type}.json')

    # Check if model config exists
    if not os.path.exists(model_config_path):
        raise ValueError(f"Model config not found: {model_config_path}")

    # Load model-specific config
    model_config = read_json_config(model_config_path)

    # Merge configs (model config takes precedence)
    merged_config = deep_update(base_config, model_config)

    return merged_config


def save_merged_config(config: Dict[str, Any], output_path: str) -> None:
    """Save merged configuration to file.

    Args:
        config: Configuration dictionary to save
        output_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)