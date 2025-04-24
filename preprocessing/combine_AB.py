#!/usr/bin/env python
"""
Combine A and B domain images side-by-side for Pix2Pix training.

This script takes input directories containing A and B domain images and combines
them side-by-side to create the aligned training data needed for Pix2Pix.
"""

import os
import argparse
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import sys
import re

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Combine A and B domain images for Pix2Pix training")

    parser.add_argument("--dir_A", type=str, required=True,
                        help="Directory containing domain A images")
    parser.add_argument("--dir_B", type=str, required=True,
                        help="Directory containing domain B images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for combined images")
    parser.add_argument("--phase", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Phase for saving the data (train, val, test)")
    parser.add_argument("--direction", type=str, default="AtoB",
                        choices=["AtoB", "BtoA"],
                        help="Which domain to put on the left side (A is first by default)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Size to resize images to before combining")
    parser.add_argument("--match_method", type=str, default="filename",
                        choices=["filename", "index", "regex", "metadata"],
                        help="Method for matching A and B images")
    parser.add_argument("--regex_pattern", type=str,
                        help="Regex pattern for extracting matching key from filenames")
    parser.add_argument("--metadata_file", type=str,
                        help="JSON file with image pairing metadata")
    parser.add_argument("--no_resize", action="store_true",
                        help="Do not resize images before combining")
    parser.add_argument("--save_unpaired", action="store_true",
                        help="Save list of unpaired images")
    return parser.parse_args()


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
        extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

    image_files = []
    for ext in extensions:
        image_files.extend(list(Path(directory).glob(f"**/*{ext}")))

    return sorted([str(f) for f in image_files])


def match_images_by_filename(files_A, files_B):
    """
    Match images from domains A and B by filename (without extension)

    Args:
        files_A: List of files from domain A
        files_B: List of files from domain B

    Returns:
        List of (domain_a_file, domain_b_file) pairs
    """
    # Create dictionaries with filenames (without extension) as keys
    domain_a_dict = {Path(f).stem: f for f in files_A}
    domain_b_dict = {Path(f).stem: f for f in files_B}

    # Find common keys
    common_keys = set(domain_a_dict.keys()).intersection(set(domain_b_dict.keys()))

    # Create pairs
    pairs = []
    unpaired_A = []
    unpaired_B = []

    for key in sorted(common_keys):
        pairs.append((domain_a_dict[key], domain_b_dict[key]))

    # Find unpaired images
    for key in domain_a_dict:
        if key not in common_keys:
            unpaired_A.append(domain_a_dict[key])

    for key in domain_b_dict:
        if key not in common_keys:
            unpaired_B.append(domain_b_dict[key])

    return pairs, unpaired_A, unpaired_B


def match_images_by_index(files_A, files_B):
    """
    Match images from domains A and B by their index in the sorted list

    Args:
        files_A: List of files from domain A
        files_B: List of files from domain B

    Returns:
        List of (domain_a_file, domain_b_file) pairs
    """
    pairs = []
    unpaired_A = []
    unpaired_B = []

    min_len = min(len(files_A), len(files_B))

    # Create pairs for common indices
    for i in range(min_len):
        pairs.append((files_A[i], files_B[i]))

    # Any extra files are unpaired
    if len(files_A) > min_len:
        unpaired_A = files_A[min_len:]

    if len(files_B) > min_len:
        unpaired_B = files_B[min_len:]

    return pairs, unpaired_A, unpaired_B


def match_images_by_regex(files_A, files_B, pattern):
    """
    Match images using a regex pattern to extract a key from filenames

    Args:
        files_A: List of files from domain A
        files_B: List of files from domain B
        pattern: Regex pattern with a capture group for the matching key

    Returns:
        List of (domain_a_file, domain_b_file) pairs
    """
    # Compile regex pattern
    regex = re.compile(pattern)

    # Extract keys from filenames
    domain_a_dict = {}
    for f in files_A:
        match = regex.search(Path(f).name)
        if match and match.groups():
            key = match.group(1)
            domain_a_dict[key] = f

    domain_b_dict = {}
    for f in files_B:
        match = regex.search(Path(f).name)
        if match and match.groups():
            key = match.group(1)
            domain_b_dict[key] = f

    # Find common keys
    common_keys = set(domain_a_dict.keys()).intersection(set(domain_b_dict.keys()))

    # Create pairs
    pairs = []
    unpaired_A = []
    unpaired_B = []

    for key in sorted(common_keys):
        pairs.append((domain_a_dict[key], domain_b_dict[key]))

    # Find unpaired images
    for key in domain_a_dict:
        if key not in common_keys:
            unpaired_A.append(domain_a_dict[key])

    for key in domain_b_dict:
        if key not in common_keys:
            unpaired_B.append(domain_b_dict[key])

    return pairs, unpaired_A, unpaired_B


def match_images_by_metadata(files_A, files_B, metadata_file):
    """
    Match images using a metadata file

    Args:
        files_A: List of files from domain A
        files_B: List of files from domain B
        metadata_file: Path to JSON file with image pairing metadata

    Returns:
        List of (domain_a_file, domain_b_file) pairs
    """
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Convert to dictionaries for faster lookup
    a_files_dict = {Path(f).name: f for f in files_A}
    b_files_dict = {Path(f).name: f for f in files_B}

    # Create pairs based on metadata
    pairs = []
    unpaired_A = []
    unpaired_B = []

    used_A = set()
    used_B = set()

    for pair in metadata:
        if 'a_file' in pair and 'b_file' in pair:
            a_name = pair['a_file']
            b_name = pair['b_file']

            if a_name in a_files_dict and b_name in b_files_dict:
                pairs.append((a_files_dict[a_name], b_files_dict[b_name]))
                used_A.add(a_name)
                used_B.add(b_name)

    # Find unpaired images
    for a_name, a_path in a_files_dict.items():
        if a_name not in used_A:
            unpaired_A.append(a_path)

    for b_name, b_path in b_files_dict.items():
        if b_name not in used_B:
            unpaired_B.append(b_path)

    return pairs, unpaired_A, unpaired_B


def combine_AB_images(files_A, files_B, output_dir, args):
    """
    Combine images from domains A and B side by side

    Args:
        files_A: List of files from domain A
        files_B: List of files from domain B
        output_dir: Directory to save combined images
        args: Command line arguments

    Returns:
        Dictionary with results info
    """
    # Select matching method
    if args.match_method == 'filename':
        pairs, unpaired_A, unpaired_B = match_images_by_filename(files_A, files_B)
    elif args.match_method == 'index':
        pairs, unpaired_A, unpaired_B = match_images_by_index(files_A, files_B)
    elif args.match_method == 'regex':
        if not args.regex_pattern:
            raise ValueError("Regex pattern must be provided for regex matching method")
        pairs, unpaired_A, unpaired_B = match_images_by_regex(files_A, files_B, args.regex_pattern)
    elif args.match_method == 'metadata':
        if not args.metadata_file:
            raise ValueError("Metadata file must be provided for metadata matching method")
        pairs, unpaired_A, unpaired_B = match_images_by_metadata(files_A, files_B, args.metadata_file)
    else:
        raise ValueError(f"Unknown matching method: {args.match_method}")

    logger.info(f"Found {len(pairs)} image pairs")
    logger.info(f"Unpaired images: {len(unpaired_A)} in domain A, {len(unpaired_B)} in domain B")

    # Create output directory
    phase_dir = Path(output_dir) / args.phase
    os.makedirs(phase_dir, exist_ok=True)

    # Process pairs
    combined_images = []

    for i, (path_A, path_B) in enumerate(tqdm(pairs, desc="Combining images")):
        # Open images
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')

        # Resize if requested
        if not args.no_resize:
            img_A = img_A.resize((args.img_size, args.img_size), Image.BICUBIC)
            img_B = img_B.resize((args.img_size, args.img_size), Image.BICUBIC)
        else:
            # Ensure same size, use A's dimensions
            if img_A.size != img_B.size:
                img_B = img_B.resize(img_A.size, Image.BICUBIC)

        # Create combined image based on direction
        width, height = img_A.size
        combined = Image.new('RGB', (width * 2, height))

        if args.direction == 'AtoB':
            combined.paste(img_A, (0, 0))
            combined.paste(img_B, (width, 0))
        else:  # BtoA
            combined.paste(img_B, (0, 0))
            combined.paste(img_A, (width, 0))

        # Save combined image
        output_path = phase_dir / f"{i:04d}.png"
        combined.save(output_path)

        combined_images.append({
            'index': i,
            'a_path': path_A,
            'b_path': path_B,
            'combined_path': str(output_path)
        })

    # Save unpaired images list if requested
    if args.save_unpaired and (unpaired_A or unpaired_B):
        unpaired_file = Path(output_dir) / f"unpaired_{args.phase}.json"
        with open(unpaired_file, 'w') as f:
            json.dump({
                'unpaired_A': unpaired_A,
                'unpaired_B': unpaired_B
            }, f, indent=2)

        logger.info(f"Saved list of unpaired images to {unpaired_file}")

    # Save metadata
    metadata_file = Path(output_dir) / f"metadata_{args.phase}.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'phase': args.phase,
            'direction': args.direction,
            'match_method': args.match_method,
            'img_size': args.img_size if not args.no_resize else None,
            'num_pairs': len(pairs),
            'num_unpaired_A': len(unpaired_A),
            'num_unpaired_B': len(unpaired_B),
            'combined_images': combined_images
        }, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")

    return {
        'num_pairs': len(pairs),
        'num_unpaired_A': len(unpaired_A),
        'num_unpaired_B': len(unpaired_B),
        'output_dir': str(phase_dir)
    }


def main():
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find images in each domain
    logger.info(f"Finding images in domain A directory: {args.dir_A}")
    files_A = find_image_files(args.dir_A)
    logger.info(f"Found {len(files_A)} images in domain A")

    logger.info(f"Finding images in domain B directory: {args.dir_B}")
    files_B = find_image_files(args.dir_B)
    logger.info(f"Found {len(files_B)} images in domain B")

    if not files_A:
        logger.error(f"No images found in domain A directory: {args.dir_A}")
        return

    if not files_B:
        logger.error(f"No images found in domain B directory: {args.dir_B}")
        return

    # Combine images
    result = combine_AB_images(files_A, files_B, args.output_dir, args)

    logger.info(f"Combined {result['num_pairs']} image pairs to {result['output_dir']}")
    if result['num_unpaired_A'] > 0 or result['num_unpaired_B'] > 0:
        logger.info(f"Unpaired images: {result['num_unpaired_A']} in domain A, {result['num_unpaired_B']} in domain B")


if __name__ == "__main__":
    main()