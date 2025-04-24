import os
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np


class ImagePair:
    """
    Class representing a pair of input and target images for image-to-image translation.
    """

    def __init__(
            self,
            input_img: np.ndarray,
            target_img: np.ndarray,
            input_path: Optional[str] = None,
            target_path: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an image pair.

        Args:
            input_img: Input image as numpy array
            target_img: Target image as numpy array
            input_path: Path to input image (optional)
            target_path: Path to target image (optional)
            metadata: Additional metadata (optional)
        """
        self.input_img = input_img
        self.target_img = target_img
        self.input_path = input_path
        self.target_path = target_path
        self.metadata = metadata or {}

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert images to PyTorch tensors.

        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        # Convert to float32
        input_float = self.input_img.astype(np.float32)
        target_float = self.target_img.astype(np.float32)

        # Normalize to [0, 1] if needed
        if input_float.max() > 1.0:
            input_float = input_float / 255.0
        if target_float.max() > 1.0:
            target_float = target_float / 255.0

        # Convert to tensors
        input_tensor = torch.from_numpy(input_float)
        target_tensor = torch.from_numpy(target_float)

        # Ensure channel dimension is first (C, H, W)
        if input_tensor.dim() == 3 and input_tensor.shape[2] in [1, 3, 4]:  # (H, W, C) format
            input_tensor = input_tensor.permute(2, 0, 1)
        elif input_tensor.dim() == 2:  # (H, W) format, add channel dim
            input_tensor = input_tensor.unsqueeze(0)

        if target_tensor.dim() == 3 and target_tensor.shape[2] in [1, 3, 4]:  # (H, W, C) format
            target_tensor = target_tensor.permute(2, 0, 1)
        elif target_tensor.dim() == 2:  # (H, W) format, add channel dim
            target_tensor = target_tensor.unsqueeze(0)

        return input_tensor, target_tensor


class ExperimentConfig:
    """
    Configuration class for experiments.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize experiment configuration.

        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict

    @classmethod
    def from_file(cls, config_path: str) -> 'ExperimentConfig':
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            ExperimentConfig object
        """
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Configuration value
        """
        keys = key.split('.')
        config = self.config

        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()


class TrainingResults:
    """
    Class for storing and tracking training results.
    """

    def __init__(self):
        """Initialize training results."""
        self.epoch_losses = []
        self.validation_metrics = []
        self.best_metric = None
        self.best_epoch = None
        self.training_time = 0.0
        self.early_stopped = False

    def add_epoch_loss(self, epoch: int, losses: Dict[str, float]) -> None:
        """
        Add loss for an epoch.

        Args:
            epoch: Epoch number
            losses: Dictionary of loss values
        """
        self.epoch_losses.append({
            'epoch': epoch,
            **losses
        })

    def add_validation_metric(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Add validation metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric values
        """
        self.validation_metrics.append({
            'epoch': epoch,
            **metrics
        })

    def update_best_metric(self, metric_name: str, metric_value: float, epoch: int,
                           mode: str = 'max') -> bool:
        """
        Update best metric if current value is better.

        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            epoch: Current epoch
            mode: 'max' if higher is better, 'min' if lower is better

        Returns:
            True if metric improved, False otherwise
        """
        improved = False

        if self.best_metric is None:
            improved = True
        elif mode == 'max' and metric_value > self.best_metric:
            improved = True
        elif mode == 'min' and metric_value < self.best_metric:
            improved = True

        if improved:
            self.best_metric = metric_value
            self.best_epoch = epoch

        return improved

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Training results as dictionary
        """
        return {
            'epoch_losses': self.epoch_losses,
            'validation_metrics': self.validation_metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'training_time': self.training_time,
            'early_stopped': self.early_stopped
        }