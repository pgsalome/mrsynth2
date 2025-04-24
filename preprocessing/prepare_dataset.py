#!/usr/bin/env python
"""
Prepare datasets for image-to-image translation models.

This script prepares datasets for training image-to-image translation models by:
1. Organizing images into appropriate directory structures
2. Splitting datasets into train/val/test
3. Creating paired or unpaired datasets for different model types
4. Preprocessing and augmenting images as needed

Supports creating datasets for:
- CycleGAN (unpaired)
- Pix2Pix (paired/aligned)
- Single domain (testing/inference)
"""

import os
import argparse
import json
import random
import shutil
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from typing import Dict, List, Any, Tuple, Optional, Union
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import ensure_dir
from preprocessing.utils.file_utils import find_image_files, match_files_by_pattern, pair_files_by_name

# Set up logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Prepare datasets for image-to-image translation models")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for prepared data")
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["pix2pix", "cyclegan", "single", "pickle"],
                        help="Type of dataset to prepare")

    # Dataset organization parameters
    parser.add_argument("--is_paired", action="store_true",
                        help="Whether the data is paired (both domains in same image or matched files)")
    parser.add_argument("--domain_a_dir", type=str,
                        help="Subdirectory containing domain A images (if not paired)")
    parser.add_argument("--domain_b_dir", type=str,
                        help="Subdirectory containing domain B images (if not paired)")
    parser.add_argument("--paired_dir", type=str,
                        help="Subdirectory containing paired images (if paired)")

    # Pairing methods
    parser.add_argument("--pair_mode", type=str, default="filename",
                        choices=["filename", "regex", "order", "substring"],
                        help="Method for pairing domain A and B images")
    parser.add_argument("--regex_pattern", type=str,
                        help="Regex pattern for extracting matching key from filenames")
    parser.add_argument("--paired_format", type=str, choices=["side_by_side", "separate"],
                        help="Format of paired data: 'side_by_side' (A|B in same image) or 'separate' (separate files)")

    # Dataset splitting parameters
    parser.add_argument("--split_ratio", type=str, default="0.7,0.15,0.15",
                        help="Train, validation, test split ratio as comma-separated values")

    # Image processing parameters
    parser.add_argument("--no_resize", action="store_true",
                        help="Do not resize images")
    parser.add_argument("--target_size", type=str, default="256,256",
                        help="Target size for resizing as height,width")
    parser.add_argument("--preprocess", type=str, default="resize_and_crop",
                        choices=["resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "none"],
                        help="Preprocessing method for images")
    parser.add_argument("--no_flip", action="store_true",
                        help="If specified, do not flip images during training")

    # Dataset-specific parameters
    parser.add_argument("--direction", type=str, default="AtoB",
                        choices=["AtoB", "BtoA"],
                        help="AtoB or BtoA, determines which domain is input")
    parser.add_argument("--load_size", type=int, default=286,
                        help="Size to load the images before cropping")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Size to crop the images")

    # Other parameters
    parser.add_argument("--copy_images", action="store_true",
                        help="Copy images instead of creating symlinks")
    parser.add_argument("--dataset_name", type=str, default="dataset",
                        help="Name of the output dataset")
    parser.add_argument("--force", action="store_true",
                        help="Force preprocessing even if output already exists")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def resize_image(img: Image.Image, size: Tuple[int, int],
                 interpolation: int = Image.BICUBIC) -> Image.Image:
    """
    Resize an image to the specified size.

    Args:
        img: PIL Image
        size: Target size as (width, height)
        interpolation: Interpolation method

    Returns:
        Resized image
    """
    return img.resize(size, interpolation)


def crop_image(img: Image.Image, pos: Tuple[int, int], size: Tuple[int, int]) -> Image.Image:
    """
    Crop an image at position pos to size.

    Args:
        img: PIL Image
        pos: Position as (x, y)
        size: Size as (width, height)

    Returns:
        Cropped image
    """
    return img.crop((pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]))


def random_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """
    Random crop an image to size.

    Args:
        img: PIL Image
        size: Size as (width, height)

    Returns:
        Randomly cropped image
    """
    w, h = img.size
    tw, th = size
    if w == tw and h == th:
        return img

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return crop_image(img, (x1, y1), (tw, th))


def scale_width(img: Image.Image, target_width: int) -> Image.Image:
    """
    Scale image to target width while maintaining aspect ratio.

    Args:
        img: PIL Image
        target_width: Target width

    Returns:
        Width-scaled image
    """
    w, h = img.size
    target_height = int(target_width * h / w)
    return img.resize((target_width, target_height), Image.BICUBIC)


def preprocess_image(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """
    Preprocess a single image based on parameters.

    Args:
        img: PIL Image
        params: Dictionary of preprocessing parameters

    Returns:
        Preprocessed PIL Image
    """
    method = params.get('preprocess', 'resize_and_crop')
    load_size = params.get('load_size', 286)
    crop_size = params.get('crop_size', 256)
    no_flip = params.get('no_flip', False)

    if method == 'resize_and_crop':
        # Resize to load_size
        if isinstance(load_size, int):
            img = resize_image(img, (load_size, load_size))
        else:
            img = resize_image(img, load_size)
        # Random crop to crop_size
        if isinstance(crop_size, int):
            img = random_crop(img, (crop_size, crop_size))
        else:
            img = random_crop(img, crop_size)
    elif method == 'crop':
        # Random crop to crop_size
        if isinstance(crop_size, int):
            img = random_crop(img, (crop_size, crop_size))
        else:
            img = random_crop(img, crop_size)
    elif method == 'scale_width':
        # Scale width to load_size
        if isinstance(load_size, int):
            img = scale_width(img, load_size)
        else:
            img = scale_width(img, load_size[0])
    elif method == 'scale_width_and_crop':
        # Scale width to load_size and crop to crop_size
        if isinstance(load_size, int):
            img = scale_width(img, load_size)
        else:
            img = scale_width(img, load_size[0])
        # Random crop to crop_size
        if isinstance(crop_size, int):
            img = random_crop(img, (crop_size, crop_size))
        else:
            img = random_crop(img, crop_size)
    elif method == 'none':
        pass
    else:
        raise ValueError(f"Preprocessing method {method} not recognized")

    # Apply flip if needed
    if not no_flip and random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img


def find_paired_images(files_a: List[str], files_b: List[str], pair_mode: str,
                       regex_pattern: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Find paired images between two domains.

    Args:
        files_a: List of files from domain A
        files_b: List of files from domain B
        pair_mode: Method for pairing ('filename', 'regex', 'order', 'substring')
        regex_pattern: Regex pattern for extracting matching key from filenames

    Returns:
        List of (domain_a_file, domain_b_file) pairs
    """
    if pair_mode == 'filename':
        pairs = pair_files_by_name(files_a, files_b)
    elif pair_mode == 'regex' and regex_pattern:
        # Match files using a regex pattern
        a_dict = match_files_by_pattern(files_a, regex_pattern)
        b_dict = match_files_by_pattern(files_b, regex_pattern)

        # Find common keys
        common_keys = set(a_dict.keys()).intersection(set(b_dict.keys()))
        pairs = [(a_dict[k], b_dict[k]) for k in sorted(common_keys)]
    elif pair_mode == 'order':
        # Pair by position in the list (requires same number of files)
        min_len = min(len(files_a), len(files_b))
        pairs = list(zip(files_a[:min_len], files_b[:min_len]))
    elif pair_mode == 'substring':
        # Pair by finding matching substrings in filenames
        pairs = []
        for a_file in files_a:
            a_name = Path(a_file).stem
            match = None

            # Find matching B file with shared substring
            for b_file in files_b:
                b_name = Path(b_file).stem
                if a_name in b_name or b_name in a_name:
                    match = b_file
                    break

            if match:
                pairs.append((a_file, match))
    else:
        raise ValueError(f"Unknown pairing mode: {pair_mode}")

    return pairs


def prepare_cyclegan_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Prepare an unpaired dataset for CycleGAN.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with dataset information
    """
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse split ratio
    split_ratio = tuple(map(float, args.split_ratio.split(',')))
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        logger.warning(f"Split ratio {split_ratio} does not sum to 1, normalizing")
        total = sum(split_ratio)
        split_ratio = tuple(r / total for r in split_ratio)

    # Create paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Find domain A and B images
    if args.domain_a_dir and args.domain_b_dir:
        domain_a_dir = input_dir / args.domain_a_dir
        domain_b_dir = input_dir / args.domain_b_dir

        if not domain_a_dir.exists():
            raise ValueError(f"Domain A directory does not exist: {domain_a_dir}")
        if not domain_b_dir.exists():
            raise ValueError(f"Domain B directory does not exist: {domain_b_dir}")

        domain_a_files = find_image_files(domain_a_dir)
        domain_b_files = find_image_files(domain_b_dir)
    else:
        # Auto-detect domains by finding directories with images
        potential_domains = []
        for dir_path in input_dir.iterdir():
            if dir_path.is_dir():
                image_files = find_image_files(dir_path)
                if image_files:
                    potential_domains.append({
                        'dir': dir_path,
                        'files': image_files
                    })

        if len(potential_domains) < 2:
            raise ValueError(f"Could not find at least 2 domains in {input_dir}")

        # Sort by number of files (descending)
        potential_domains.sort(key=lambda d: len(d['files']), reverse=True)

        # Use the two domains with the most files
        domain_a_dir = potential_domains[0]['dir']
        domain_a_files = potential_domains[0]['files']

        domain_b_dir = potential_domains[1]['dir']
        domain_b_files = potential_domains[1]['files']

    logger.info(f"Found {len(domain_a_files)} images in domain A and {len(domain_b_files)} images in domain B")

    # Shuffle files
    random.shuffle(domain_a_files)
    random.shuffle(domain_b_files)

    # Split domain A files
    train_a_size = int(len(domain_a_files) * split_ratio[0])
    val_a_size = int(len(domain_a_files) * split_ratio[1])

    train_a_files = domain_a_files[:train_a_size]
    val_a_files = domain_a_files[train_a_size:train_a_size + val_a_size]
    test_a_files = domain_a_files[train_a_size + val_a_size:]

    # Split domain B files
    train_b_size = int(len(domain_b_files) * split_ratio[0])
    val_b_size = int(len(domain_b_files) * split_ratio[1])

    train_b_files = domain_b_files[:train_b_size]
    val_b_files = domain_b_files[train_b_size:train_b_size + val_b_size]
    test_b_files = domain_b_files[train_b_size + val_b_size:]

    # Create output directories
    train_a_dir = output_dir / "trainA"
    train_b_dir = output_dir / "trainB"
    val_a_dir = output_dir / "valA"
    val_b_dir = output_dir / "valB"
    test_a_dir = output_dir / "testA"
    test_b_dir = output_dir / "testB"

    ensure_dir(train_a_dir)
    ensure_dir(train_b_dir)
    ensure_dir(val_a_dir)
    ensure_dir(val_b_dir)
    ensure_dir(test_a_dir)
    ensure_dir(test_b_dir)

    # Parse target size
    if not args.no_resize:
        target_size = tuple(map(int, args.target_size.split(',')))
    else:
        target_size = None

    # Process and save images
    processed = {
        'trainA': process_unaligned_files(train_a_files, train_a_dir, target_size, args),
        'trainB': process_unaligned_files(train_b_files, train_b_dir, target_size, args),
        'valA': process_unaligned_files(val_a_files, val_a_dir, target_size, args),
        'valB': process_unaligned_files(val_b_files, val_b_dir, target_size, args),
        'testA': process_unaligned_files(test_a_files, test_a_dir, target_size, args),
        'testB': process_unaligned_files(test_b_files, test_b_dir, target_size, args)
    }

    # Save dataset info
    dataset_info = {
        'dataset_type': 'cyclegan',
        'direction': args.direction,
        'split_ratio': split_ratio,
        'target_size': target_size,
        'domain_a_dir': str(domain_a_dir),
        'domain_b_dir': str(domain_b_dir),
        'num_files': {
            'trainA': len(train_a_files),
            'trainB': len(train_b_files),
            'valA': len(val_a_files),
            'valB': len(val_b_files),
            'testA': len(test_a_files),
            'testB': len(test_b_files)
        },
        'processed': processed
    }

    # Save metadata
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"CycleGAN dataset preparation complete. Results saved to {output_dir}")
    return dataset_info


def process_unaligned_files(files: List[str], output_dir: Path, target_size: Optional[Tuple[int, int]],
                            args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Process and save unaligned image files.

    Args:
        files: List of image files
        output_dir: Output directory
        target_size: Target size for resizing (width, height)
        args: Command-line arguments

    Returns:
        List of processed files information
    """
    processed = []

    for i, file_path in enumerate(tqdm(files, desc=f"Processing {output_dir.name}")):
        # Load image
        try:
            img = Image.open(file_path).convert('RGB')

            # Resize if needed
            if target_size and not args.no_resize:
                img = img.resize(target_size, Image.BICUBIC)

            # Save image
            output_filename = f"{i:04d}.png"
            output_path = output_dir / output_filename

            img.save(output_path)

            processed.append({
                'id': i,
                'original_path': file_path,
                'processed_path': str(output_path)
            })
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")

    return processed


def prepare_pix2pix_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Prepare a paired dataset for Pix2Pix.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with dataset information
    """
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse split ratio
    split_ratio = tuple(map(float, args.split_ratio.split(',')))
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        logger.warning(f"Split ratio {split_ratio} does not sum to 1, normalizing")
        total = sum(split_ratio)
        split_ratio = tuple(r / total for r in split_ratio)

    # Create paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Check how the data is organized
    pairs = []

    if args.paired_format == 'side_by_side':
        # Images are already side by side (A|B)
        if args.paired_dir:
            paired_dir = input_dir / args.paired_dir

            if not paired_dir.exists():
                raise ValueError(f"Paired directory does not exist: {paired_dir}")

            # Find all paired images
            paired_files = find_image_files(paired_dir)
            logger.info(f"Found {len(paired_files)} paired images")

            # No need to pair, just use the files directly
            pairs = [(f, f) for f in paired_files]  # Use same file for both A and B
    else:
        # Images are in separate domains, need to pair them
        if not args.domain_a_dir or not args.domain_b_dir:
            raise ValueError("Both --domain_a_dir and --domain_b_dir must be provided for separate files")

        domain_a_dir = input_dir / args.domain_a_dir
        domain_b_dir = input_dir / args.domain_b_dir

        if not domain_a_dir.exists():
            raise ValueError(f"Domain A directory does not exist: {domain_a_dir}")
        if not domain_b_dir.exists():
            raise ValueError(f"Domain B directory does not exist: {domain_b_dir}")

        # Find images in each domain
        domain_a_files = find_image_files(domain_a_dir)
        domain_b_files = find_image_files(domain_b_dir)

        logger.info(f"Found {len(domain_a_files)} images in domain A and {len(domain_b_files)} images in domain B")

        # Pair images from the two domains
        pairs = find_paired_images(domain_a_files, domain_b_files, args.pair_mode, args.regex_pattern)

        if not pairs:
            raise ValueError("No paired images found")

        logger.info(f"Found {len(pairs)} paired images")

    # Shuffle pairs
    random.shuffle(pairs)

    # Split into train, val, and test
    train_size = int(len(pairs) * split_ratio[0])
    val_size = int(len(pairs) * split_ratio[1])

    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:train_size + val_size]
    test_pairs = pairs[train_size + val_size:]

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    ensure_dir(train_dir)
    ensure_dir(val_dir)
    ensure_dir(test_dir)

    # Parse target size
    if not args.no_resize:
        target_height, target_width = map(int, args.target_size.split(','))
        # For pix2pix, we need double width to accommodate A|B
        if args.paired_format == 'separate':
            target_size = (target_width, target_height)
        else:
            # For side_by_side, the width is already doubled
            target_size = (target_width * 2, target_height)
    else:
        target_size = None

    # Process and save images
    processed = {
        'train': process_pix2pix_pairs(train_pairs, train_dir, target_size, args),
        'val': process_pix2pix_pairs(val_pairs, val_dir, target_size, args),
        'test': process_pix2pix_pairs(test_pairs, test_dir, target_size, args)
    }

    # Save dataset info
    dataset_info = {
        'dataset_type': 'pix2pix',
        'direction': args.direction,
        'paired_format': args.paired_format,
        'split_ratio': split_ratio,
        'target_size': target_size,
        'pair_mode': args.pair_mode if args.paired_format == 'separate' else None,
        'domain_a_dir': str(args.domain_a_dir) if args.domain_a_dir else None,
        'domain_b_dir': str(args.domain_b_dir) if args.domain_b_dir else None,
        'paired_dir': str(args.paired_dir) if args.paired_dir else None,
        'num_pairs': {
            'train': len(train_pairs),
            'val': len(val_pairs),
            'test': len(test_pairs)
        },
        'processed': processed
    }

    # Save metadata
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Pix2Pix dataset preparation complete. Results saved to {output_dir}")
    return dataset_info


def process_pix2pix_pairs(pairs: List[Tuple[str, str]], output_dir: Path,
                          target_size: Optional[Tuple[int, int]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Process and save paired images for Pix2Pix.

    Args:
        pairs: List of paired image files
        output_dir: Output directory
        target_size: Target size for resizing
        args: Command-line arguments

    Returns:
        List of processed pairs information
    """
    processed = []

    for i, (path_a, path_b) in enumerate(tqdm(pairs, desc=f"Processing {output_dir.name}")):
        try:
            # For side_by_side format, both paths point to the same file
            if args.paired_format == 'side_by_side':
                # Load combined image
                img = Image.open(path_a).convert('RGB')

                # Resize if needed
                if target_size and not args.no_resize:
                    img = img.resize(target_size, Image.BICUBIC)

                # Save directly
                output_filename = f"{i:04d}.png"
                output_path = output_dir / output_filename

                img.save(output_path)

                processed.append({
                    'id': i,
                    'original_path': path_a,
                    'processed_path': str(output_path)
                })

            else:  # separate format
                # Load individual images
                img_a = Image.open(path_a).convert('RGB')
                img_b = Image.open(path_b).convert('RGB')

                # Resize if needed
                if target_size and not args.no_resize:
                    img_a = img_a.resize(target_size, Image.BICUBIC)
                    img_b = img_b.resize(target_size, Image.BICUBIC)
                else:
                    # Ensure same size by resizing B to match A
                    if img_a.size != img_b.size:
                        img_b = img_b.resize(img_a.size, Image.BICUBIC)

                # Combine side by side
                width, height = img_a.size
                combined = Image.new('RGB', (width * 2, height))

                # Put A on the left and B on the right (or vice versa depending on direction)
                if args.direction == 'AtoB':
                    combined.paste(img_a, (0, 0))
                    combined.paste(img_b, (width, 0))
                else:  # BtoA
                    combined.paste(img_b, (0, 0))
                    combined.paste(img_a, (width, 0))

                # Save combined image
                output_filename = f"{i:04d}.png"
                output_path = output_dir / output_filename

                combined.save(output_path)

                processed.append({
                    'id': i,
                    'original_path_a': path_a,
                    'original_path_b': path_b,
                    'processed_path': str(output_path)
                })

        except Exception as e:
            logger.warning(f"Error processing {path_a} and {path_b}: {e}")

    return processed


def prepare_single_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Prepare a single-domain dataset for testing/inference.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with dataset information
    """
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Find images in input directory
    if args.domain_a_dir:
        domain_dir = input_dir / args.domain_a_dir
    else:
        domain_dir = input_dir

    if not domain_dir.exists():
        raise ValueError(f"Input directory does not exist: {domain_dir}")

    image_files = find_image_files(domain_dir)
    logger.info(f"Found {len(image_files)} images in {domain_dir}")

    # Create output directory
    ensure_dir(output_dir)

    # Parse target size
    if not args.no_resize:
        target_size = tuple(map(int, args.target_size.split(',')))
    else:
        target_size = None

    # Process and save images
    processed = process_unaligned_files(image_files, output_dir, target_size, args)

    # Save dataset info
    dataset_info = {
        'dataset_type': 'single',
        'target_size': target_size,
        'num_images': len(processed),
        'domain_dir': str(domain_dir),
        'processed': processed
    }

    # Save metadata
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Single domain dataset preparation complete. Results saved to {output_dir}")
    return dataset_info


def prepare_pickle_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Prepare a pickle dataset with ImagePair objects.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with dataset information
    """
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse split ratio
    split_ratio = tuple(map(float, args.split_ratio.split(',')))
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        logger.warning(f"Split ratio {split_ratio} does not sum to 1, normalizing")
        total = sum(split_ratio)
        split_ratio = tuple(r / total for r in split_ratio)

    # Create paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Ensure output directory exists
    ensure_dir(output_dir)

    # Find domain A and B images
    if not args.domain_a_dir or not args.domain_b_dir:
        raise ValueError("Both --domain_a_dir and --domain_b_dir must be provided for pickle dataset")

    domain_a_dir = input_dir / args.domain_a_dir
    domain_b_dir = input_dir / args.domain_b_dir

    if not domain_a_dir.exists():
        raise ValueError(f"Domain A directory does not exist: {domain_a_dir}")
    if not domain_b_dir.exists():
        raise ValueError(f"Domain B directory does not exist: {domain_b_dir}")

    # Find images in each domain
    domain_a_files = find_image_files(domain_a_dir)
    domain_b_files = find_image_files(domain_b_dir)

    logger.info(f"Found {len(domain_a_files)} images in domain A and {len(domain_b_files)} images in domain B")

    # Pair images from the two domains
    pairs = find_paired_images(domain_a_files, domain_b_files, args.pair_mode, args.regex_pattern)

    if not pairs:
        raise ValueError("No paired images found")

    logger.info(f"Found {len(pairs)} paired images")

    # Create ImagePair objects
    try:
        from utils.dataclasses import ImagePair

        # Create output pickle file path
        output_path = output_dir / f"{args.dataset_name}.pkl"

        # Create ImagePair objects
        logger.info("Creating ImagePair objects...")
        image_pairs = []

        for a_file, b_file in tqdm(pairs, desc="Creating ImagePair objects"):
            try:
                # Load images
                img_a = cv2.imread(a_file)
                if img_a is None:
                    raise ValueError(f"Failed to load image: {a_file}")
                img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

                img_b = cv2.imread(b_file)
                if img_b is None:
                    raise ValueError(f"Failed to load image: {b_file}")
                img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

                # Ensure both images have the same size
                if img_a.shape[:2] != img_b.shape[:2]:
                    img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

                # Create ImagePair object
                pair = ImagePair(
                    input_img=img_a,
                    target_img=img_b,
                    input_path=a_file,
                    target_path=b_file
                )

                image_pairs.append(pair)
            except Exception as e:
                logger.warning(f"Error processing {a_file} and {b_file}: {e}")

        # Shuffle and split dataset
        random.shuffle(image_pairs)

        train_size = int(len(image_pairs) * split_ratio[0])
        val_size = int(len(image_pairs) * split_ratio[1])
        test_size = len(image_pairs) - train_size - val_size

        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, len(image_pairs)))

        # Save pickle dataset
        logger.info(f"Saving {len(image_pairs)} image pairs to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(image_pairs, f)

        # Save dataset info
        dataset_info = {
            'dataset_type': 'pickle',
            'pickle_path': str(output_path),
            'num_pairs': len(image_pairs),
            'splits': {
                'train': {'indices': train_indices, 'size': len(train_indices)},
                'val': {'indices': val_indices, 'size': len(val_indices)},
                'test': {'indices': test_indices, 'size': len(test_indices)}
            },
            'domain_a_dir': str(domain_a_dir),
            'domain_b_dir': str(domain_b_dir),
            'pair_mode': args.pair_mode,
            'split_ratio': split_ratio
        }

        # Save metadata
        with open(output_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        logger.info(f"Pickle dataset preparation complete. Results saved to {output_dir}")
        return dataset_info

    except ImportError:
        logger.error("ImagePair class not found. Make sure utils.dataclasses is available.")
        raise


def main():
    """Main function"""
    args = parse_args()

    # Create output directory
    ensure_dir(args.output_dir)

    # Prepare dataset based on type
    if args.dataset_type == "cyclegan":
        prepare_cyclegan_dataset(args)
    elif args.dataset_type == "pix2pix":
        prepare_pix2pix_dataset(args)
    elif args.dataset_type == "single":
        prepare_single_dataset(args)
    elif args.dataset_type == "pickle":
        prepare_pickle_dataset(args)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")


if __name__ == "__main__":
    main()