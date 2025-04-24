from typing import Dict, Any, List
import torch
from torch.utils.data import DataLoader, Dataset


def create_dataloader(dataset: Dataset, config: Dict[str, Any], shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for a dataset.

    Args:
        dataset: Dataset to create loader for
        config: Configuration dictionary
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader for the dataset
    """
    # Extract dataloader parameters from config
    batch_size = config.get('batch_size', 1)
    num_workers = config.get('num_workers', 4)
    pin_memory = config.get('pin_memory', True)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    return dataloader


def create_dataloaders(datasets: Dict[str, Dataset], config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for datasets.

    Args:
        datasets: Dictionary mapping split names to datasets
        config: Configuration dictionary

    Returns:
        Dictionary mapping split names to dataloaders
    """
    dataloaders = {}

    # Extract dataloader parameters from config
    data_config = config.get('data', {})

    # Create dataloader for each dataset
    for split, dataset in datasets.items():
        # Only shuffle training data
        shuffle = split == 'train' and data_config.get('shuffle', True)

        # Create dataloader
        dataloaders[split] = create_dataloader(dataset, data_config, shuffle)

    return dataloaders