import os
import torch
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import random
from PIL import Image

from src.utils.dataclass import ImageTranslationDataset
from src.utils.io import load_pickle


def prepare_transforms(config: Dict[str, Any]) -> Dict[str, transforms.Compose]:
    """
    Prepare image transformations for training, validation, and test sets.

    Args:
        config: Configuration dictionary containing image size and augmentation parameters.

    Returns:
        Dictionary with train, val, and test transform pipelines.
    """
    img_size = config["data"]["img_size"]
    load_size = int(img_size * 1.1)  # Make load size a bit larger for random cropping
    no_flip = config["data"].get("no_flip", False)

    # Training transforms with augmentation
    train_transforms = []
    train_transforms.append(
        transforms.Resize((load_size, load_size), interpolation=transforms.InterpolationMode.BICUBIC))
    train_transforms.append(transforms.RandomCrop((img_size, img_size)))
    if not no_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    train_transform = transforms.Compose(train_transforms)

    # Validation/test transforms without augmentation
    val_transforms = []
    val_transforms.append(transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC))
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    val_transform = transforms.Compose(val_transforms)

    return {
        "train": train_transform,
        "val": val_transform,
        "test": val_transform
    }


def create_datasets(config: Dict[str, Any]) -> Dict[str, Dataset]:
    """
    Create training, validation, and test datasets.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary mapping split names to dataset objects
    """
    # Get image directories
    data_dir = Path(config["data"]["dataset_dir"])
    dataset_mode = config["data"].get("dataset_mode", "aligned")
    direction = config["data"].get("direction", "AtoB")

    # Prepare transformations
    transforms_dict = prepare_transforms(config)

    # Check if using existing preprocessed dataset pickle
    if config["data"].get("dataset_name", "").endswith(".pkl") and os.path.exists(
            data_dir / config["data"]["dataset_name"]):
        print(f"Loading preprocessed dataset from {data_dir / config['data']['dataset_name']}")
        dataset = load_pickle(str(data_dir / config["data"]["dataset_name"]))

        # Split dataset
        train_ratio = config["data"]["train_split"]
        val_ratio = config["data"]["val_split"]
        test_ratio = config["data"]["test_split"]

        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

        # Split dataset
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config["seed"])
        )

        # Create dataset objects
        train_dataset = ImageTranslationDataset(train_dataset, transform=transforms_dict["train"])
        val_dataset = ImageTranslationDataset(val_dataset, transform=transforms_dict["val"])
        test_dataset = ImageTranslationDataset(test_dataset, transform=transforms_dict["test"])

    else:
        # Create dataset based on dataset_mode
        if dataset_mode == 'aligned':
            # For aligned datasets (pix2pix.json)
            train_dataset = AlignedDataset(
                root=data_dir,
                phase='train',
                transform=transforms_dict["train"],
                direction=direction
            )

            val_dataset = AlignedDataset(
                root=data_dir,
                phase='val',
                transform=transforms_dict["val"],
                direction=direction
            )

            test_dataset = AlignedDataset(
                root=data_dir,
                phase='test',
                transform=transforms_dict["test"],
                direction=direction
            )

        elif dataset_mode == 'unaligned':
            # For unaligned datasets (CycleGAN)
            train_dataset = UnalignedDataset(
                root=data_dir,
                phase='train',
                transform=transforms_dict["train"],
                direction=direction
            )

            val_dataset = UnalignedDataset(
                root=data_dir,
                phase='val',
                transform=transforms_dict["val"],
                direction=direction
            )

            test_dataset = UnalignedDataset(
                root=data_dir,
                phase='test',
                transform=transforms_dict["test"],
                direction=direction
            )

        elif dataset_mode == 'single':
            # For inference on single images
            train_dataset = None
            val_dataset = None

            test_dataset = SingleDataset(
                root=data_dir,
                transform=transforms_dict["test"],
                direction=direction
            )

        else:
            raise ValueError(f"Dataset mode {dataset_mode} not supported")

    datasets = {}
    if train_dataset is not None:
        datasets["train"] = train_dataset
    if val_dataset is not None:
        datasets["val"] = val_dataset
    if test_dataset is not None:
        datasets["test"] = test_dataset

    # Print dataset sizes
    print(f"Dataset sizes:")
    for split, dataset in datasets.items():
        print(f"  {split}: {len(dataset)}")

    return datasets


def create_dataloaders(datasets: Dict[str, Dataset], config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Create dataloaders for datasets.

    Args:
        datasets: Dictionary mapping split names to dataset objects
        config: Configuration dictionary

    Returns:
        Dictionary mapping split names to dataloader objects
    """
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    shuffle = config["data"]["shuffle"]

    dataloaders = {}

    for split, dataset in datasets.items():
        # Only shuffle training data
        should_shuffle = shuffle if split == "train" else False

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    return dataloaders


class AlignedDataset(Dataset):
    """
    Dataset class for aligned image domains (pix2pix.json).

    This dataset assumes that the images in domain A and B are aligned and stored as side-by-side images.
    """

    def __init__(self, root, phase='train', transform=None, direction='AtoB', max_dataset_size=float("inf")):
        """
        Initialize the dataset.

        Args:
            root: Root directory containing the images
            phase: 'train', 'val', or 'test'
            transform: Image transformations to apply
            direction: 'AtoB' or 'BtoA' - which side of the image is the input
            max_dataset_size: Maximum number of images to use
        """
        self.root = Path(root)
        self.phase = phase
        self.transform = transform
        self.direction = direction
        self.max_dataset_size = max_dataset_size

        # Find the image directory
        self.dir = self.root / phase
        if not self.dir.exists():
            raise ValueError(f"Directory {self.dir} does not exist")

        # Get all image paths
        self.image_paths = sorted([p for p in self.dir.glob('*.jpg') if p.is_file()])
        self.image_paths.extend(sorted([p for p in self.dir.glob('*.png') if p.is_file()]))

        # Limit dataset size
        self.image_paths = self.image_paths[:min(len(self.image_paths), int(max_dataset_size))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - an image in the input domain
            B (tensor) - its corresponding image in the target domain
            A_paths (str) - image paths
            B_paths (str) - image paths
        """
        # Read image
        path = str(self.image_paths[index])
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


class UnalignedDataset(Dataset):
    """
    Dataset class for unaligned image domains (CycleGAN).

    This dataset assumes that the images in domain A and B are not aligned and stored in separate directories.
    """

    def __init__(self, root, phase='train', transform=None, direction='AtoB', max_dataset_size=float("inf")):
        """
        Initialize the dataset.

        Args:
            root: Root directory containing the images
            phase: 'train', 'val', or 'test'
            transform: Image transformations to apply
            direction: 'AtoB' or 'BtoA' - which domain is the input
            max_dataset_size: Maximum number of images to use
        """
        self.root = Path(root)
        self.phase = phase
        self.transform = transform
        self.direction = direction
        self.max_dataset_size = max_dataset_size

        # Find the domain directories
        self.dir_A = self.root / f'{phase}A'
        self.dir_B = self.root / f'{phase}B'

        # Create alternative paths if standard paths don't exist
        if not self.dir_A.exists() or not self.dir_B.exists():
            self.dir_A = self.root / 'A'
            self.dir_B = self.root / 'B'

        if not self.dir_A.exists() or not self.dir_B.exists():
            raise ValueError(f"Directories {self.dir_A} and {self.dir_B} do not exist")

        # Get all image paths
        self.A_paths = sorted([str(p) for p in self.dir_A.glob('*.jpg') if p.is_file()])
        self.A_paths.extend(sorted([str(p) for p in self.dir_A.glob('*.png') if p.is_file()]))

        self.B_paths = sorted([str(p) for p in self.dir_B.glob('*.jpg') if p.is_file()])
        self.B_paths.extend(sorted([str(p) for p in self.dir_B.glob('*.png') if p.is_file()]))

        # Limit dataset size
        self.A_size = min(len(self.A_paths), int(max_dataset_size))
        self.B_size = min(len(self.B_paths), int(max_dataset_size))

        self.A_paths = self.A_paths[:self.A_size]
        self.B_paths = self.B_paths[:self.B_size]

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - an image in the input domain
            B (tensor) - its corresponding image in the target domain
            A_paths (str) - image paths
            B_paths (str) - image paths
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


class SingleDataset(Dataset):
    """Dataset class for testing on a single-domain dataset (inference only)."""

    def __init__(self, root, transform=None, direction='AtoB', max_dataset_size=float("inf")):
        """
        Initialize the dataset.

        Args:
            root: Root directory containing the images
            transform: Image transformations to apply
            direction: 'AtoB' or 'BtoA' - which domain is the input
            max_dataset_size: Maximum number of images to use
        """
        self.root = Path(root)
        self.transform = transform
        self.direction = direction
        self.max_dataset_size = max_dataset_size

        # Find the domain directory (only need A for inference)
        self.dir = self.root / 'test' if (self.root / 'test').exists() else self.root

        # Get all image paths
        self.paths = sorted([str(p) for p in self.dir.glob('*.jpg') if p.is_file()])
        self.paths.extend(sorted([str(p) for p in self.dir.glob('*.png') if p.is_file()]))

        # Limit dataset size
        self.size = min(len(self.paths), int(max_dataset_size))
        self.paths = self.paths[:self.size]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A (tensor) - an image in the input domain
            A_paths (str) - image paths
        """
        # Read image
        path = self.paths[index]
        img = Image.open(path).convert('RGB')

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return {'A': img, 'A_paths': path}