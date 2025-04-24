#!/usr/bin/env python
"""
Data preparation for image-to-image translation models.

This script prepares datasets for training image-to-image translation models by:
1. Organizing images into appropriate directory structures
2. Splitting datasets into train/val/test
3. Creating metadata for tracking dataset provenance
4. Optionally preprocessing and augmenting the data
"""

import os
import argparse
import logging
from pathlib import Path
import json
import random
import numpy as np
from typing import Dict, List, Any, Tuple
from PIL import Image
from tqdm import tqdm
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import ensure_dir

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Prepare data for image-to-image translation models")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for prepared data")
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["pix2pix", "cyclegan", "single"],
                        help="Type of dataset to prepare")
    parser.add_argument("--is_paired", action="store_true",
                        help="Whether the data is paired (both domains in same image or matched files)")
    parser.add_argument("--domain_a_dir", type=str,
                        help="Subdirectory containing domain A images (if not paired)")
    parser.add_argument("--domain_b_dir", type=str,
                        help="Subdirectory containing domain B images (if not paired)")
    parser.add_argument("--paired_dir", type=str,
                        help="Subdirectory containing paired images (if paired)")
    parser.add_argument("--paired_format", type=str, choices=["side_by_side", "separate"],
                        help="Format of paired data: 'side_by_side' (A|B in same image) or 'separate' (separate files)")
    parser.add_argument("--split_ratio", type=str, default="0.7,0.15,0.15",
                        help="Train, validation, test split ratio as comma-separated values")
    parser.add_argument("--no_resize", action="store_true",
                        help="Do not resize images")
    parser.add_argument("--target_size", type=str, default="256,256",
                        help="Target size for resizing as height,width")
    parser.add_argument("--copy_images", action="store_true",
                        help="Copy images instead of creating symlinks")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset shuffling and splitting")
    parser.add_argument("--preprocess", action="store_true",
                        help="Apply preprocessing (normalization)")

    return parser.parse_args()


def find_image_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Find all image files in a directory.

    Args:
        directory: Directory to search
        recursive: Whether to search recursively

    Returns:
        List of image file paths
    """
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    image_files = []

    directory_path = Path(directory)
    if recursive:
        for ext in extensions:
            image_files.extend([str(p) for p in directory_path.glob(f"**/*{ext}")])
    else:
        for ext in extensions:
            image_files.extend([str(p) for p in directory_path.glob(f"*{ext}")])

    return sorted(image_files)


def split_dataset(files: List[str], split_ratio: Tuple[float, float, float]) -> Dict[str, List[str]]:
    """
    Split a list of files into train, validation, and test sets.

    Args:
        files: List of file paths
        split_ratio: (train_ratio, val_ratio, test_ratio) tuple

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to file lists
    """
    # Ensure ratios sum to 1
    train_ratio, val_ratio, test_ratio = split_ratio
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    # Shuffle files
    shuffled_files = files.copy()
    random.shuffle(shuffled_files)

    # Calculate split sizes
    n_files = len(shuffled_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    n_test = n_files - n_train - n_val

    # Split files
    train_files = shuffled_files[:n_train]
    val_files = shuffled_files[n_train:n_train + n_val]
    test_files = shuffled_files[n_train + n_val:]

    return {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }


def prepare_cyclegan_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Prepare a CycleGAN dataset.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with dataset information
    """
    # Validate inputs
    if args.is_paired:
        raise ValueError("CycleGAN requires unpaired data. Use --is_paired=False.")

    if not args.domain_a_dir or not args.domain_b_dir:
        raise ValueError("Both --domain_a_dir and --domain_b_dir must be provided for CycleGAN.")

    # Create paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    domain_a_dir = input_dir / args.domain_a_dir
    domain_b_dir = input_dir / args.domain_b_dir

    # Check directories
    if not domain_a_dir.exists():
        raise ValueError(f"Domain A directory does not exist: {domain_a_dir}")
    if not domain_b_dir.exists():
        raise ValueError(f"Domain B directory does not exist: {domain_b_dir}")

    # Find images
    domain_a_files = find_image_files(domain_a_dir)
    domain_b_files = find_image_files(domain_b_dir)

    logger.info(f"Found {len(domain_a_files)} images in domain A and {len(domain_b_files)} images in domain B")

    # Create split ratios
    split_ratio = tuple(map(float, args.split_ratio.split(',')))

    # Split datasets
    domain_a_splits = split_dataset(domain_a_files, split_ratio)
    domain_b_splits = split_dataset(domain_b_files, split_ratio)

    # Create output directories
    for phase in ['train', 'val', 'test']:
        ensure_dir(output_dir / f"{phase}A")
        ensure_dir(output_dir / f"{phase}B")

    # Parse target size
    if not args.no_resize:
        target_size = tuple(map(int, args.target_size.split(',')))
    else:
        target_size = None

    # Process and organize images
    processed_images = {
        'A': {'train': [], 'val': [], 'test': []},
        'B': {'train': [], 'val': [], 'test': []}
    }

    # Process domain A
    for phase in ['train', 'val', 'test']:
        logger.info(f"Processing domain A {phase} set ({len(domain_a_splits[phase])} images)")

        for i, src_path in enumerate(tqdm(domain_a_splits[phase])):
            # Create destination path
            dest_filename = f"{i:04d}.png"
            dest_path = output_dir / f"{phase}A" / dest_filename

            # Process image
            img = Image.open(src_path).convert('RGB')

            # Resize if needed
            if target_size and not args.no_resize:
                img = img.resize(target_size, Image.BICUBIC)

            # Save image
            img.save(dest_path)

            # Record metadata
            processed_images['A'][phase].append({
                'original_path': src_path,
                'processed_path': str(dest_path),
                'size': img.size
            })

    # Process domain B
    for phase in ['train', 'val', 'test']:
        logger.info(f"Processing domain B {phase} set ({len(domain_b_splits[phase])} images)")

        for i, src_path in enumerate(tqdm(domain_b_splits[phase])):
            # Create destination path
            dest_filename = f"{i:04d}.png"
            dest_path = output_dir / f"{phase}B" / dest_filename

            # Process image
            img = Image.open(src_path).convert('RGB')

            # Resize if needed
            if target_size and not args.no_resize:
                img = img.resize(target_size, Image.BICUBIC)

            # Save image
            img.save(dest_path)

            # Record metadata
            processed_images['B'][phase].append({
                'original_path': src_path,
                'processed_path': str(dest_path),
                'size': img.size
            })

    # Save dataset info
    dataset_info = {
        'dataset_type': 'cyclegan',
        'split_ratio': split_ratio,
        'target_size': target_size,
        'num_images': {
            'A': {
                'train': len(processed_images['A']['train']),
                'val': len(processed_images['A']['val']),
                'test': len(processed_images['A']['test'])
            },
            'B': {
                'train': len(processed_images['B']['train']),
                'val': len(processed_images['B']['val']),
                'test': len(processed_images['B']['test'])
            }
        },
        'processed_images': processed_images
    }

    # Save metadata
    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Dataset preparation complete. Results saved to {output_dir}")

    return dataset_info


def prepare_pix2pix_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Prepare a Pix2Pix dataset.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with dataset information
    """
    # Validate inputs
    if not args.is_paired:
        raise ValueError("Pix2Pix requires paired data. Use --is_paired=True.")

    # Create paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Check paired format
    if args.paired_format == "side_by_side":
        # Images are already side by side (A|B)
        if not args.paired_dir:
            raise ValueError("--paired_dir must be provided for side_by_side format.")

        paired_dir = input_dir / args.paired_dir
        if not paired_dir.exists():
            raise ValueError(f"Paired directory does not exist: {paired_dir}")

        # Find images
        paired_files = find_image_files(paired_dir)
        logger.info(f"Found {len(paired_files)} paired images")

        # Create split ratios
        split_ratio = tuple(map(float, args.split_ratio.split(',')))

        # Split dataset
        splits = split_dataset(paired_files, split_ratio)

        # Create output directories
        for phase in ['train', 'val', 'test']:
            ensure_dir(output_dir / phase)

        # Process and organize images
        processed_images = {
            'train': [], 'val': [], 'test': []
        }

        # Parse target size
        if not args.no_resize:
            target_height, target_width = map(int, args.target_size.split(','))
            target_size = (target_width * 2, target_height)  # Double width for A|B
        else:
            target_size = None

        # Process images
        for phase in ['train', 'val', 'test']:
            logger.info(f"Processing {phase} set ({len(splits[phase])} images)")

            for i, src_path in enumerate(tqdm(splits[phase])):
                # Create destination path
                dest_filename = f"{i:04d}.png"
                dest_path = output_dir / phase / dest_filename

                # Process image
                img = Image.open(src_path).convert('RGB')

                # Resize if needed
                if target_size and not args.no_resize:
                    img = img.resize(target_size, Image.BICUBIC)

                # Save image
                img.save(dest_path)

                # Record metadata
                processed_images[phase].append({
                    'original_path': src_path,
                    'processed_path': str(dest_path),
                    'size': img.size
                })

    elif args.paired_format == "separate":
        # Images are in separate files but paired
        if not args.domain_a_dir or not args.domain_b_dir:
            raise ValueError("Both --domain_a_dir and --domain_b_dir must be provided for separate format.")

        domain_a_dir = input_dir / args.domain_a_dir
        domain_b_dir = input_dir / args.domain_b_dir

        # Check directories
        if not domain_a_dir.exists():
            raise ValueError(f"Domain A directory does not exist: {domain_a_dir}")
        if not domain_b_dir.exists():
            raise ValueError(f"Domain B directory does not exist: {domain_b_dir}")

        # Find images
        domain_a_files = find_image_files(domain_a_dir)
        domain_b_files = find_image_files(domain_b_dir)

        logger.info(f"Found {len(domain_a_files)} images in domain A and {len(domain_b_files)} images in domain B")

        # Match files by name (without extension)
        a_dict = {Path(f).stem: f for f in domain_a_files}
        b_dict = {Path(f).stem: f for f in domain_b_files}
        common_keys = set(a_dict.keys()).intersection(set(b_dict.keys()))

        logger.info(f"Found {len(common_keys)} matching image pairs")

        if not common_keys:
            raise ValueError("No matching image pairs found. Check filenames.")

        # Create paired files
        paired_files = [(a_dict[k], b_dict[k]) for k in common_keys]

        # Create split ratios
        split_ratio = tuple(map(float, args.split_ratio.split(',')))

        # Shuffle and split pairs
        random.shuffle(paired_files)
        n_pairs = len(paired_files)
        n_train = int(n_pairs * split_ratio[0] / sum(split_ratio))
        n_val = int(n_pairs * split_ratio[1] / sum(split_ratio))
        n_test = n_pairs - n_train - n_val

        train_pairs = paired_files[:n_train]
        val_pairs = paired_files[n_train:n_train + n_val]
        test_pairs = paired_files[n_train + n_val:]

        splits = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }

        # Create output directories
        for phase in ['train', 'val', 'test']:
            ensure_dir(output_dir / phase)

        # Process and organize images
        processed_images = {
            'train': [], 'val': [], 'test': []
        }

        # Parse target size
        if not args.no_resize:
            target_height, target_width = map(int, args.target_size.split(','))
            target_size = (target_width, target_height)
        else:
            target_size = None

        # Process images
        for phase in ['train', 'val', 'test']:
            logger.info(f"Processing {phase} set ({len(splits[phase])} image pairs)")

            for i, (path_a, path_b) in enumerate(tqdm(splits[phase])):
                # Create destination path
                dest_filename = f"{i:04d}.png"
                dest_path = output_dir / phase / dest_filename

                # Load images
                img_a = Image.open(path_a).convert('RGB')
                img_b = Image.open(path_b).convert('RGB')

                # Resize if needed
                if target_size and not args.no_resize:
                    img_a = img_a.resize(target_size, Image.BICUBIC)
                    img_b = img_b.resize(target_size, Image.BICUBIC)
                else:
                    # Ensure same size, resize B to match A
                    if img_a.size != img_b.size:
                        img_b = img_b.resize(img_a.size, Image.BICUBIC)

                # Combine images side by side
                width, height = img_a.size
                combined = Image.new('RGB', (width * 2, height))
                combined.paste(img_a, (0, 0))
                combined.paste(img_b, (width, 0))

                # Save combined image
                combined.save(dest_path)

                # Record metadata
                processed_images[phase].append({
                    'original_path_a': path_a,
                    'original_path_b': path_b,
                    'processed_path': str(dest_path),
                    'size': combined.size
                })

    else:
        raise ValueError(f"Unknown paired format: {args.paired_format}")

    # Save dataset info
    dataset_info = {
        'dataset_type': 'pix2pix',
        'paired_format': args.paired_format,
        'split_ratio': split_ratio,
        'target_size': target_size,
        'num_images': {
            'train': len(processed_images['train']),
            'val': len(processed_images['val']),
            'test': len(processed_images['test'])
        },
        'processed_images': processed_images
    }

    # Save metadata
    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Dataset preparation complete. Results saved to {output_dir}")

    return dataset_info


def prepare_single_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Prepare a single-domain dataset for testing.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with dataset information
    """
    # Create paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Find all images
    image_files = find_image_files(input_dir)
    logger.info(f"Found {len(image_files)} images")

    # Create output directory
    ensure_dir(output_dir)

    # Parse target size
    if not args.no_resize:
        target_size = tuple(map(int, args.target_size.split(',')))
    else:
        target_size = None

    # Process images
    processed_images = []

    for i, src_path in enumerate(tqdm(image_files, desc="Processing images")):
        # Create destination path
        dest_filename = f"{i:04d}.png"
        dest_path = output_dir / dest_filename

        # Process image
        img = Image.open(src_path).convert('RGB')

        # Resize if needed
        if target_size and not args.no_resize:
            img = img.resize(target_size, Image.BICUBIC)

        # Save image
        img.save(dest_path)

        # Record metadata
        processed_images.append({
            'original_path': src_path,
            'processed_path': str(dest_path),
            'size': img.size
        })

    # Save dataset info
    dataset_info = {
        'dataset_type': 'single',
        'target_size': target_size,
        'num_images': len(processed_images),
        'processed_images': processed_images
    }

    # Save metadata
    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Dataset preparation complete. Results saved to {output_dir}")

    return dataset_info


def main():
    # Parse arguments
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare dataset based on type
    if args.dataset_type == "cyclegan":
        prepare_cyclegan_dataset(args)
    elif args.dataset_type == "pix2pix":
        prepare_pix2pix_dataset(args)
    elif args.dataset_type == "single":
        prepare_single_dataset(args)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")


if __name__ == "__main__":
    main()