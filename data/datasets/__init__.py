"""Dataset exports and factory functions."""

from typing import Dict, Any, Optional
from torch.utils.data import Dataset
import os

from .aligned_dataset import AlignedDataset
from .unaligned_dataset import UnalignedDataset
from .single_dataset import SingleDataset


class ImageTranslationDataset(Dataset):
    """
    Dataset wrapper for image translation datasets loaded from processed data.
    """

    def __init__(self, dataset_items, transform=None):
        """
        Initialize the dataset.

        Args:
            dataset_items: List of dataset items (ImagePair objects or similar)
            transform: Transformations to apply to images
        """
        self.dataset_items = dataset_items
        self.transform = transform

    def __len__(self):
        """Get dataset size."""
        return len(self.dataset_items)

    def __getitem__(self, index):
        """
        Get a data point.

        Args:
            index: Index of the data point

        Returns:
            Dictionary with data
        """
        item = self.dataset_items[index]

        # Handle ImagePair objects
        if hasattr(item, 'input_img') and hasattr(item, 'target_img'):
            input_tensor, target_tensor = item.to_tensors()

            # Apply transformations if provided
            if self.transform:
                input_tensor = self.transform(input_tensor)
                target_tensor = self.transform(target_tensor)

            return {
                'A': input_tensor,
                'B': target_tensor,
                'A_paths': item.input_path if hasattr(item, 'input_path') else None,
                'B_paths': item.target_path if hasattr(item, 'target_path') else None
            }

        # Handle dictionary-like items
        elif isinstance(item, dict):
            result = {}
            for key, value in item.items():
                if hasattr(value, 'shape') and self.transform:
                    result[key] = self.transform(value)
                else:
                    result[key] = value
            return result

        # Handle other types
        else:
            if self.transform:
                item = self.transform(item)
            return item


def create_dataset(config: Dict[str, Any], phase: str = 'train') -> Dataset:
    """
    Create a dataset based on configuration.

    Args:
        config: Configuration dictionary
        phase: Dataset phase ('train', 'val', or 'test')

    Returns:
        Dataset object
    """
    # Extract dataset parameters from config
    dataset_mode = config.get('dataset_mode', 'aligned')
    dataset_dir = config.get('dataset_dir', './datasets')
    direction = config.get('direction', 'AtoB')
    max_dataset_size = config.get('max_dataset_size', float('inf'))
    load_size = config.get('load_size', 286)
    crop_size = config.get('crop_size', 256)
    preprocess = config.get('preprocess', 'resize_and_crop')
    no_flip = config.get('no_flip', False)

    # Generate the dataset configuration
    dataset_config = {
        'dataroot': dataset_dir,
        'phase': phase,
        'direction': direction,
        'max_dataset_size': max_dataset_size,
        'load_size': load_size,
        'crop_size': crop_size,
        'preprocess': preprocess,
        'no_flip': no_flip
    }

    # Create dataset based on mode
    if dataset_mode == 'aligned':
        dataset = AlignedDataset(**dataset_config)
    elif dataset_mode == 'unaligned':
        dataset = UnalignedDataset(**dataset_config)
    elif dataset_mode == 'single':
        dataset = SingleDataset(**dataset_config)
    else:
        raise ValueError(f"Dataset mode {dataset_mode} not recognized")

    return dataset


def create_datasets(config: Dict[str, Any]) -> Dict[str, Dataset]:
    """
    Create training, validation, and test datasets.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary mapping phases to datasets
    """
    datasets = {}

    # Create training dataset
    if not config.get('test_mode', False):
        datasets['train'] = create_dataset(config, 'train')

        # Create validation dataset if enabled
        if config.get('enable_validation', True):
            datasets['val'] = create_dataset(config, 'val')

    # Create test dataset
    datasets['test'] = create_dataset(config, 'test')

    return datasets