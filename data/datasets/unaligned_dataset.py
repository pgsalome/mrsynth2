import os
import random
from pathlib import Path
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from typing import Callable, Optional, Dict, Any


class UnalignedDataset(data.Dataset):
    """
    Dataset class for unpaired image-to-image translation (CycleGAN).

    This dataset assumes that the images in domain A and B are unaligned
    and stored in separate directories.
    """

    def __init__(
            self,
            dataroot: str,
            phase: str = 'train',
            direction: str = 'AtoB',
            transform: Optional[Callable] = None,
            load_size: int = 286,
            crop_size: int = 256,
            preprocess: str = 'resize_and_crop',
            no_flip: bool = False,
            max_dataset_size: float = float("inf")
    ):
        """
        Initialize the dataset.

        Args:
            dataroot: Root directory containing the data
            phase: Train/val/test phase
            direction: AtoB or BtoA, determines which domain is input
            transform: Optional transform to apply
            load_size: Size to load the images
            crop_size: Size to crop the images
            preprocess: Preprocessing method
            no_flip: If True, don't flip images
            max_dataset_size: Maximum number of samples to load
        """
        self.dataroot = Path(dataroot)
        self.phase = phase
        self.direction = direction

        # Load size parameters
        self.load_size = load_size
        self.crop_size = crop_size
        self.preprocess = preprocess
        self.no_flip = no_flip

        # Find the domain directories
        self.dir_A = self.dataroot / f'{phase}A'
        self.dir_B = self.dataroot / f'{phase}B'

        # Create alternative paths if standard paths don't exist
        if not self.dir_A.exists() or not self.dir_B.exists():
            self.dir_A = self.dataroot / 'A'
            self.dir_B = self.dataroot / 'B'

        if not self.dir_A.exists() or not self.dir_B.exists():
            raise ValueError(f"Directories {self.dir_A} and {self.dir_B} do not exist")

        # Get all image paths
        self.A_paths = sorted([str(p) for p in self.dir_A.glob('*.jpg')])
        self.A_paths.extend(sorted([str(p) for p in self.dir_A.glob('*.png')]))

        self.B_paths = sorted([str(p) for p in self.dir_B.glob('*.jpg')])
        self.B_paths.extend(sorted([str(p) for p in self.dir_B.glob('*.png')]))

        # Limit dataset size
        self.A_size = min(len(self.A_paths), int(max_dataset_size))
        self.B_size = min(len(self.B_paths), int(max_dataset_size))

        self.A_paths = self.A_paths[:self.A_size]
        self.B_paths = self.B_paths[:self.B_size]

        # Create transform if not provided
        self.transform = transform
        if self.transform is None:
            self.transform = self._get_transform()

    def __len__(self):
        """Get dataset size."""
        return max(self.A_size, self.B_size)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a data point.

        Args:
            index: Index of the data point

        Returns:
            Dictionary containing A, B, A_paths, and B_paths
        """
        # Make sure index is within range
        A_index = index % self.A_size
        B_index = random.randint(0, self.B_size - 1)  # Random B image

        # Read images
        A_path = self.A_paths[A_index]
        B_path = self.B_paths[B_index]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply transformations
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

        # Swap A and B if direction is BtoA
        if self.direction == 'BtoA':
            A_img, B_img = B_img, A_img
            A_path, B_path = B_path, A_path

        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

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
            if self.phase == 'train':
                transform_list.append(transforms.RandomCrop(self.crop_size))
            else:
                transform_list.append(transforms.CenterCrop(self.crop_size))

        # Flip
        if self.phase == 'train' and not self.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return transforms.Compose(transform_list)