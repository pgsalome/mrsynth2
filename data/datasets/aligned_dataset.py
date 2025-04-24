"""Dataset for aligned image pairs (Pix2Pix)."""

from pathlib import Path
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from typing import Callable, Optional, Dict, Any


class AlignedDataset(data.Dataset):
    """
    Dataset class for paired image-to-image translation (Pix2Pix).

    This dataset assumes that the images in domain A and B are aligned and
    stored as side-by-side images or matched files.
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

        # Find the image directory
        self.dir = self.dataroot / phase
        if not self.dir.exists():
            raise ValueError(f"Directory {self.dir} does not exist")

        # Get all image paths
        self.image_paths = sorted([str(p) for p in self.dir.glob('*.jpg')])
        self.image_paths.extend(sorted([str(p) for p in self.dir.glob('*.png')]))

        # Limit dataset size
        self.image_paths = self.image_paths[:min(len(self.image_paths), int(max_dataset_size))]

        # Create transform if not provided
        self.transform = transform
        if self.transform is None:
            self.transform = self._get_transform()

    def __len__(self):
        """Get dataset size."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a data point.

        Args:
            index: Index of the data point

        Returns:
            Dictionary containing A, B, A_paths, and B_paths
        """
        # Read image
        path = self.image_paths[index]
        img = Image.open(path).convert('RGB')

        # Split A and B images
        w, h = img.size
        w2 = int(w / 2)

        if self.direction == 'AtoB':
            img_A = img.crop((0, 0, w2, h))
            img_B = img.crop((w2, 0, w, h))
        else:
            img_A = img.crop((w2, 0, w, h))
            img_B = img.crop((0, 0, w2, h))

        # Apply transformations
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B, 'A_paths': path, 'B_paths': path}

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