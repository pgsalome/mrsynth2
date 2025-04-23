#!/usr/bin/env python
"""
Create final datasets for image-to-image translation models

This script organizes processed MRI slices into datasets suitable for
different image-to-image translation models (CycleGAN, Pix2Pix, etc.)
It handles dataset splitting, copying files to the right directories,
and creating metadata for training.
"""

import os
import argparse
import logging
import json
import random
import shutil
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

from src.utils.dataclass import ImagePair

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_image_files(directory, extensions=None):
    """
    Find all image files in a directory

    Args:
        directory: Directory to search
        extensions: List of valid extensions (default: ['.png', '.jpg', '.jpeg'])

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg']

    image_files = []
    for ext in extensions:
        image_files.extend(list(Path(directory).glob(f"**/*{ext}")))

    return [str(f) for f in image_files]


def load_image(image_path):
    """
    Load an image file

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array
    """
    try:
        # Try loading with PIL first
        img = Image.open(image_path)
        img_array = np.array(img)

        # Convert grayscale to RGB if needed
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=2)
        elif img_array.shape[2] == 1:
            img_array = np.concatenate([img_array] * 3, axis=2)

        return img_array
    except Exception as e:
        # Try loading with OpenCV
        try:
            img_array = cv2.imread(image_path)
            if img_array is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Convert BGR to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            return img_array
        except Exception as e2:
            raise ValueError(f"Failed to load image: {image_path}, {e}, {e2}")


def find_paired_images(domain_a_files, domain_b_files, mode='filename'):
    """
    Find paired images between two domains

    Args:
        domain_a_files: List of files from domain A
        domain_b_files: List of files from domain B
        mode: Pairing mode ('filename' or 'position')

    Returns:
        List of (domain_a_file, domain_b_file) pairs
    """
    pairs = []

    if mode == 'filename':
        # Create dictionaries with filenames (without extension) as keys
        domain_a_dict = {Path(f).stem: f for f in domain_a_files}
        domain_b_dict = {Path(f).stem: f for f in domain_b_files}

        # Find common keys
        common_keys = set(domain_a_dict.keys()).intersection(set(domain_b_dict.keys()))

        # Create pairs
        for key in common_keys:
            pairs.append((domain_a_dict[key], domain_b_dict[key]))

    elif mode == 'position':
        # Pair by position in the list (requires same number of files)
        if len(domain_a_files) != len(domain_b_files):
            logger.warning(
                f"Domain A ({len(domain_a_files)} files) and Domain B ({len(domain_b_files)} files) have different sizes")
            min_len = min(len(domain_a_files), len(domain_b_files))
            domain_a_files = domain_a_files[:min_len]
            domain_b_files = domain_b_files[:min_len]

        pairs = list(zip(domain_a_files, domain_b_files))

    elif mode == 'substring':
        # For each file in domain A, find a matching file in domain B with the same substring
        # (e.g., subject ID, slice number)
        for a_file in domain_a_files:
            a_stem = Path(a_file).stem

            # Try to find a matching file in domain B
            match = None
            for b_file in domain_b_files:
                b_stem = Path(b_file).stem

                # Check if the strings share a common part
                if a_stem in b_stem or b_stem in a_stem:
                    match = b_file
                    break

            if match:
                pairs.append((a_file, match))

    else:
        raise ValueError(f"Unknown pairing mode: {mode}")

    return pairs


def create_pix2pix_dataset(pairs, output_dir, split_ratio=(0.7, 0.15, 0.15), mode='copy'):
    """
    Create a pix2pix dataset from paired images

    Args:
        pairs: List of (domain_a_file, domain_b_file) pairs
        output_dir: Output directory
        split_ratio: (train, val, test) split ratios
        mode: 'copy' to copy files, 'combined' to create combined A|B images

    Returns:
        Dictionary with information about the created dataset
    """
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # Shuffle pairs
    random.shuffle(pairs)

    # Calculate split sizes
    train_size = int(len(pairs) * split_ratio[0])
    val_size = int(len(pairs) * split_ratio[1])
    test_size = len(pairs) - train_size - val_size

    # Split pairs
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:train_size + val_size]
    test_pairs = pairs[train_size + val_size:]

    # Process pairs
    datasets = {
        'train': {'pairs': train_pairs, 'dir': train_dir, 'files': []},
        'val': {'pairs': val_pairs, 'dir': val_dir, 'files': []},
        'test': {'pairs': test_pairs, 'dir': test_dir, 'files': []}
    }

    for split, split_data in datasets.items():
        logger.info(f"Creating {split} dataset with {len(split_data['pairs'])} pairs")

        for i, (a_file, b_file) in enumerate(tqdm(split_data['pairs'], desc=f"Processing {split} pairs")):
            if mode == 'copy':
                # Copy files to output directory
                a_dest = os.path.join(split_data['dir'], f"{i:04d}_A.png")
                b_dest = os.path.join(split_data['dir'], f"{i:04d}_B.png")

                shutil.copy(a_file, a_dest)
                shutil.copy(b_file, b_dest)

                split_data['files'].append({
                    'domain_a': a_dest,
                    'domain_b': b_dest
                })

            elif mode == 'combined':
                # Create combined A|B image for pix2pix
                a_img = load_image(a_file)
                b_img = load_image(b_file)

                # Ensure both images have the same size
                if a_img.shape != b_img.shape:
                    logger.warning(f"Images have different shapes: {a_img.shape} vs {b_img.shape}")

                    # Resize to match the first image
                    b_img = cv2.resize(b_img, (a_img.shape[1], a_img.shape[0]))

                # Create combined image (A|B)
                combined = np.concatenate([a_img, b_img], axis=1)

                # Save combined image
                dest = os.path.join(split_data['dir'], f"{i:04d}.png")
                cv2.imwrite(dest, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

                split_data['files'].append({
                    'combined': dest,
                    'domain_a': a_file,
                    'domain_b': b_file
                })

    return datasets


def create_cyclegan_dataset(domain_a_files, domain_b_files, output_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Create a CycleGAN dataset from unpaired images

    Args:
        domain_a_files: List of files from domain A
        domain_b_files: List of files from domain B
        output_dir: Output directory
        split_ratio: (train, val, test) split ratios

    Returns:
        Dictionary with information about the created dataset
    """
    # Create output directories
    splits = ['train', 'val', 'test']
    domains = ['A', 'B']

    for split in splits:
        for domain in domains:
            os.makedirs(os.path.join(output_dir, f"{split}{domain}"), exist_ok=True)

    # Shuffle files
    random.shuffle(domain_a_files)
    random.shuffle(domain_b_files)

    # Calculate split sizes for domain A
    train_a_size = int(len(domain_a_files) * split_ratio[0])
    val_a_size = int(len(domain_a_files) * split_ratio[1])
    test_a_size = len(domain_a_files) - train_a_size - val_a_size

    # Calculate split sizes for domain B
    train_b_size = int(len(domain_b_files) * split_ratio[0])
    val_b_size = int(len(domain_b_files) * split_ratio[1])
    test_b_size = len(domain_b_files) - train_b_size - val_b_size

    # Split files
    train_a = domain_a_files[:train_a_size]
    val_a = domain_a_files[train_a_size:train_a_size + val_a_size]
    test_a = domain_a_files[train_a_size + val_a_size:]

    train_b = domain_b_files[:train_b_size]
    val_b = domain_b_files[train_b_size:train_b_size + val_b_size]
    test_b = domain_b_files[train_b_size + val_b_size:]

    # Process files
    datasets = {
        'train': {
            'A': {'files': train_a, 'dir': os.path.join(output_dir, 'trainA'), 'copied_files': []},
            'B': {'files': train_b, 'dir': os.path.join(output_dir, 'trainB'), 'copied_files': []}
        },
        'val': {
            'A': {'files': val_a, 'dir': os.path.join(output_dir, 'valA'), 'copied_files': []},
            'B': {'files': val_b, 'dir': os.path.join(output_dir, 'valB'), 'copied_files': []}
        },
        'test': {
            'A': {'files': test_a, 'dir': os.path.join(output_dir, 'testA'), 'copied_files': []},
            'B': {'files': test_b, 'dir': os.path.join(output_dir, 'testB'), 'copied_files': []}
        }
    }

    for split, domains_data in datasets.items():
        for domain, domain_data in domains_data.items():
            logger.info(f"Creating {split}{domain} dataset with {len(domain_data['files'])} files")

            for i, file_path in enumerate(tqdm(domain_data['files'], desc=f"Processing {split}{domain} files")):
                # Copy file to output directory
                dest = os.path.join(domain_data['dir'], f"{i:04d}.png")
                shutil.copy(file_path, dest)

                domain_data['copied_files'].append({
                    'original': file_path,
                    'copied': dest
                })

    return datasets


def create_pickle_dataset(pairs, output_path, split_ratio=(0.7, 0.15, 0.15)):
    """
    Create a pickle dataset with ImagePair objects

    Args:
        pairs: List of (domain_a_file, domain_b_file) pairs
        output_path: Output pickle file path
        split_ratio: (train, val, test) split ratios

    Returns:
        Dictionary with information about the created dataset
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Shuffle pairs
    random.shuffle(pairs)

    # Calculate split sizes
    train_size = int(len(pairs) * split_ratio[0])
    val_size = int(len(pairs) * split_ratio[1])
    test_size = len(pairs) - train_size - val_size

    # Split pairs
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:train_size + val_size]
    test_pairs = pairs[train_size + val_size:]

    # Create ImagePair objects
    logger.info("Loading images and creating ImagePair objects")

    image_pairs = []
    for a_file, b_file in tqdm(pairs, desc="Creating ImagePair objects"):
        try:
            a_img = load_image(a_file)
            b_img = load_image(b_file)

            # Ensure both images have the same size
            if a_img.shape[:2] != b_img.shape[:2]:
                logger.warning(f"Images have different shapes: {a_img.shape} vs {b_img.shape}")

                # Resize to match the first image
                b_img = cv2.resize(b_img, (a_img.shape[1], a_img.shape[0]))

            # Create ImagePair object
            pair = ImagePair(
                input_img=a_img,
                target_img=b_img,
                input_path=a_file,
                target_path=b_file
            )

            image_pairs.append(pair)
        except Exception as e:
            logger.error(f"Error processing pair {a_file}, {b_file}: {e}")

    # Save dataset
    logger.info(f"Saving {len(image_pairs)} image pairs to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(image_pairs, f)

    # Get split indices
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, len(image_pairs)))

    return {
        'pickle_path': output_path,
        'num_pairs': len(image_pairs),
        'splits': {
            'train': {'indices': train_indices, 'size': len(train_indices)},
            'val': {'indices': val_indices, 'size': len(val_indices)},
            'test': {'indices': test_indices, 'size': len(test_indices)}
        }
    }


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Create datasets for image-to-image translation models")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing processed slices")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for dataset")
    parser.add_argument("--domain_a_dir", type=str,
                        help="Directory containing domain A images (relative to input_dir)")
    parser.add_argument("--domain_b_dir", type=str,
                        help="Directory containing domain B images (relative to input_dir)")
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["pix2pix", "cyclegan", "pickle"],
                        help="Type of dataset to create")
    parser.add_argument("--pair_mode", type=str, default="filename",
                        choices=["filename", "position", "substring"],
                        help="Method for pairing domain A and B images")
    parser.add_argument("--split_ratio", type=str, default="0.7,0.15,0.15",
                        help="Train, validation, test split ratios as comma-separated values")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dataset_name", type=str, default="dataset",
                        help="Name of the output dataset")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse split ratio
    split_ratio = tuple(map(float, args.split_ratio.split(',')))
    if len(split_ratio) != 3 or sum(split_ratio) != 1.0:
        logger.warning(f"Split ratio {split_ratio} does not sum to 1, normalizing")
        total = sum(split_ratio)
        split_ratio = tuple(r / total for r in split_ratio)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find domain A and B images
    if args.domain_a_dir and args.domain_b_dir:
        domain_a_dir = os.path.join(args.input_dir, args.domain_a_dir)
        domain_b_dir = os.path.join(args.input_dir, args.domain_b_dir)

        domain_a_files = find_image_files(domain_a_dir)
        domain_b_files = find_image_files(domain_b_dir)

        logger.info(f"Found {len(domain_a_files)} files in domain A and {len(domain_b_files)} files in domain B")
    else:
        # Auto-detect domains based on directory structure
        # Look for directories with slices
        potential_domains = []

        for root, dirs, files in os.walk(args.input_dir):
            # Check if this directory contains image files
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                potential_domains.append({
                    'dir': root,
                    'files': [os.path.join(root, f) for f in image_files]
                })

        if len(potential_domains) < 2:
            logger.error(f"Could not find at least 2 domains in {args.input_dir}")
            exit(1)

        # Sort domains by number of files (descending)
        potential_domains.sort(key=lambda d: len(d['files']), reverse=True)

        # Use the two domains with the most files
        domain_a_dir = potential_domains[0]['dir']
        domain_a_files = potential_domains[0]['files']

        domain_b_dir = potential_domains[1]['dir']
        domain_b_files = potential_domains[1]['files']

        logger.info(f"Auto-detected domains:")
        logger.info(f"  Domain A: {domain_a_dir} ({len(domain_a_files)} files)")
        logger.info(f"  Domain B: {domain_b_dir} ({len(domain_b_files)} files)")

    # Create dataset based on type
    if args.dataset_type == "pix2pix":
        logger.info("Creating Pix2Pix dataset")

        # Find paired images
        pairs = find_paired_images(domain_a_files, domain_b_files, args.pair_mode)
        logger.info(f"Found {len(pairs)} image pairs")

        # Create Pix2Pix dataset
        dataset_info = create_pix2pix_dataset(
            pairs,
            args.output_dir,
            split_ratio=split_ratio,
            mode='combined'
        )

        # Save dataset info
        with open(os.path.join(args.output_dir, 'dataset_info.json'), 'w') as f:
            json.dump({
                'dataset_type': args.dataset_type,
                'input_dir': args.input_dir,
                'domain_a_dir': domain_a_dir,
                'domain_b_dir': domain_b_dir,
                'pair_mode': args.pair_mode,
                'split_ratio': split_ratio,
                'seed': args.seed,
                'dataset_name': args.dataset_name,
                'dataset_info': {
                    'train_size': len(dataset_info['train']['files']),
                    'val_size': len(dataset_info['val']['files']),
                    'test_size': len(dataset_info['test']['files'])
                }
            }, f, indent=2)

    elif args.dataset_type == "cyclegan":
        logger.info("Creating CycleGAN dataset")

        # Create CycleGAN dataset
        dataset_info = create_cyclegan_dataset(
            domain_a_files,
            domain_b_files,
            args.output_dir,
            split_ratio=split_ratio
        )

        # Save dataset info
        with open(os.path.join(args.output_dir, 'dataset_info.json'), 'w') as f:
            json.dump({
                'dataset_type': args.dataset_type,
                'input_dir': args.input_dir,
                'domain_a_dir': domain_a_dir,
                'domain_b_dir': domain_b_dir,
                'split_ratio': split_ratio,
                'seed': args.seed,
                'dataset_name': args.dataset_name,
                'dataset_info': {
                    'train_A_size': len(dataset_info['train']['A']['files']),
                    'train_B_size': len(dataset_info['train']['B']['files']),
                    'val_A_size': len(dataset_info['val']['A']['files']),
                    'val_B_size': len(dataset_info['val']['B']['files']),
                    'test_A_size': len(dataset_info['test']['A']['files']),
                    'test_B_size': len(dataset_info['test']['B']['files'])
                }
            }, f, indent=2)

    elif args.dataset_type == "pickle":
        logger.info("Creating pickle dataset")

        # Find paired images
        pairs = find_paired_images(domain_a_files, domain_b_files, args.pair_mode)
        logger.info(f"Found {len(pairs)} image pairs")

        # Create pickle dataset
        output_path = os.path.join(args.output_dir, f"{args.dataset_name}.pkl")
        dataset_info = create_pickle_dataset(
            pairs,
            output_path,
            split_ratio=split_ratio
        )

        # Save dataset info
        with open(os.path.join(args.output_dir, 'dataset_info.json'), 'w') as f:
            json.dump({
                'dataset_type': args.dataset_type,
                'input_dir': args.input_dir,
                'domain_a_dir': domain_a_dir,
                'domain_b_dir': domain_b_dir,
                'pair_mode': args.pair_mode,
                'split_ratio': split_ratio,
                'seed': args.seed,
                'dataset_name': args.dataset_name,
                'dataset_info': dataset_info
            }, f, indent=2)

    logger.info(f"Dataset creation completed. Results saved to {args.output_dir}")