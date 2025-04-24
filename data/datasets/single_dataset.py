import os
from pathlib import Path
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from typing import Callable, Optional, Dict, Any


class SingleDataset(data.Dataset):
    """
    Dataset class for testing on a single domain (inference).

    This dataset loads images from a single directory for testing or inference.
    """

    def __init__(
            self,
            dataroot: str,
            transform: Optional[Callable] = None,
            load_size: int = 256,
            crop_size: int = 256,
            preprocess: str = 'resize',
            max_dataset_size: float = float("inf")
    ):
        """
        Initialize the dataset.

        Args:
            dataroot: Root directory containing the data
            transform: Optional transform to apply
            load_size: Size to load the images
            crop_size: Size to crop the images
            preprocess: Preprocessing method
            max_dataset_size: Maximum number of samples to load
        """
        self.dataroot = Path(dataroot)

        # Load size parameters
        self.load_size = load_size
        self.crop_size = crop_size
        self.preprocess = preprocess

        # Find the image directory
        self.dir = self.dataroot
        if not self.dir.exists():
            raise ValueError(f"Directory {self.dir} does not exist")

        # Get all image paths
        self.paths = sorted([str(p) for p in self.dir.glob('*.jpg')])
        self.paths.extend(sorted([str(p) for p in self.dir.glob('*.png')]))

        # Limit dataset size
        self.paths = self.paths[:min(len(self.paths), int(max_dataset_size))]

        # Create transform if not provided
        self.transform = transform
        if self.transform is None:
            self.transform = self._get_transform()

    def __len__(self):
        """Get dataset size."""
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a data point.

        Args:
            index: Index of the data point

        Returns:
            Dictionary containing A and A_paths
        """
        # Read image
        path = self.paths[index]
        img = Image.open(path).convert('RGB')

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return {'A': img, 'A_paths': path}

    def _get_transform(self) -> Callable:
        """
        Create transform based on preprocessing type.

        Returns:
            Transform function
        """
        transform_list = []

        # Resize
        if 'resize' in self.preprocess:
            transform_list.append(
                transforms.Resize([self.load_size, self.load_size],
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            )

        # Crop
        if 'crop' in self.preprocess:
            transform_list.append(transforms.CenterCrop(self.crop_size))

        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return transforms.Compose(transform_list)