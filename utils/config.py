import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


class Config:
    """
    Centralized configuration management system that handles loading, merging,
    and providing access to configuration values.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with optional starting dictionary.

        Args:
            config_dict: Initial configuration dictionary
        """
        self._config = config_dict or {}

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """
        Create a Config object from a JSON or YAML file.

        Args:
            config_path: Path to configuration file (JSON or YAML)

        Returns:
            Config object with loaded configuration
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix.lower() in ('.json', '.jsn'):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ('.yaml', '.yml'):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

        return cls(config_dict)

    @classmethod
    def from_args(cls, args: Union[argparse.Namespace, Dict[str, Any]]) -> 'Config':
        """
        Create a Config object from command-line arguments.

        Args:
            args: Command-line arguments (argparse.Namespace or dict)

        Returns:
            Config object with arguments as configuration
        """
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        return cls(args)

    def merge(self, other: Union[Dict[str, Any], 'Config']) -> 'Config':
        """
        Merge this configuration with another configuration.

        Args:
            other: Another Config object or dictionary to merge with

        Returns:
            New Config object with merged configurations
        """
        if isinstance(other, Config):
            other_dict = other.to_dict()
        else:
            other_dict = other

        merged = self._deep_merge(self._config, other_dict)
        return Config(merged)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config to dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a file.

        Args:
            path: Path to save configuration
        """
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)

        if path.suffix.lower() in ('.json', '.jsn'):
            with open(path, 'w') as f:
                json.dump(self._config, f, indent=2)
        elif path.suffix.lower() in ('.yaml', '.yml'):
            with open(path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: Key to get (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.

        Args:
            key: Key to set (supports dot notation for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with multiple key-value pairs.

        Args:
            updates: Dictionary of updates to apply
        """
        for key, value in updates.items():
            self.set(key, value)

    def __getitem__(self, key: str) -> Any:
        """
        Access configuration values using dictionary syntax.

        Args:
            key: Key to get

        Returns:
            Configuration value

        Raises:
            KeyError: If key not found
        """
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set configuration values using dictionary syntax.

        Args:
            key: Key to set
            value: Value to set
        """
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """
        Check if key exists in configuration.

        Args:
            key: Key to check

        Returns:
            True if key exists, False otherwise
        """
        return self.get(key) is not None

    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries.

        Args:
            base: Base dictionary
            update: Dictionary to merge on top of base

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


def load_model_config(model_type: str, base_config_path: Optional[str] = None) -> Config:
    """
    Load and merge base and model-specific configurations.

    Args:
        model_type: Type of model (e.g., 'cyclegan', 'pix2pix', 'diffusion')
        base_config_path: Path to base config file (defaults to 'config/defaults/base.json')

    Returns:
        Merged configuration
    """
    # Set default base config path if not provided
    if base_config_path is None:
        base_config_path = os.path.join('config', 'defaults', 'base.json')

    # Load base config
    base_config = Config.from_file(base_config_path)

    # Construct model config path
    model_config_path = os.path.join('config', 'defaults', f'{model_type}.json')

    # Check if model config exists
    if not os.path.exists(model_config_path):
        raise ValueError(f"Model config not found: {model_config_path}")

    # Load model-specific config
    model_config = Config.from_file(model_config_path)

    # Merge configs (model config takes precedence)
    merged_config = base_config.merge(model_config)

    return merged_config


def parse_args_and_config(parser: Optional[argparse.ArgumentParser] = None) -> Config:
    """
    Parse command-line arguments and merge with appropriate configuration files.

    Args:
        parser: Optional pre-configured argument parser

    Returns:
        Merged configuration
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="MRSynth2")

    # Add common arguments
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model_type', type=str, help='Type of model')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs to use')
    parser.add_argument('--seed', type=int, help='Random seed')

    args = parser.parse_args()

    # If config file is provided, use it as the base
    if args.config:
        config = Config.from_file(args.config)
    # Otherwise, use model_type to load appropriate configs
    elif args.model_type:
        config = load_model_config(args.model_type)
    # If neither is provided, use default base config
    else:
        base_config_path = os.path.join('config', 'defaults', 'base.json')
        config = Config.from_file(base_config_path)

    # Create args config and merge with loaded config (args take precedence)
    args_config = Config.from_args(args)
    merged_config = config.merge(args_config)

    return merged_config