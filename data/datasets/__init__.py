from typing import Dict, Any, Optional
from torch.utils.data import Dataset
import os

from .aligned_dataset import AlignedDataset
from .unaligned_dataset import UnalignedDataset
from .single_dataset import SingleDataset


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