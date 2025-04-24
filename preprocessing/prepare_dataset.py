import os
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

from preprocessing.utils.file_utils import find_image_files, pair_files_by_name, match_files_by_pattern, ensure_dir


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prepare datasets for image-to-image translation models")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for prepared data")
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["aligned", "unaligned", "single"],
                        help="Type of dataset to prepare")
    parser.add_argument("--direction", type=str, default="AtoB",
                        choices=["AtoB", "BtoA"],
                        help="Direction for translation")
    parser.add_argument("--domain_a_dir", type=str,
                        help="Directory containing domain A images")
    parser.add_argument("--domain_b_dir", type=str,
                        help="Directory containing domain B images")
    parser.add_argument("--pair_mode", type=str, default="filename",
                        choices=["filename", "regex", "order"],
                        help="Method for pairing images")
    parser.add_argument("--regex_pattern", type=str,
                        help="Regex pattern for pairing by regex")
    parser.add_argument("--split_ratio", type=str, default="0.7,0.15,0.15",
                        help="Train, validation, test split ratio as comma-separated values")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Target image size")
    parser.add_argument("--no_resize", action="store_true",
                        help="Do not resize images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def prepare_aligned_dataset(args):
    """
    Prepare an aligned dataset (e.g., for Pix2Pix).

    Args:
        args: Command line arguments

    Returns:
        Dictionary with dataset information
    """
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse split ratio
    split_ratio = tuple(map(float, args.split_ratio.split(',')))
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        print(f"Warning: Split ratio {split_ratio} does not sum to 1, normalizing")
        total = sum(split_ratio)
        split_ratio = tuple(r / total for r in split_ratio)

    # Find domain A and B images
    domain_a_dir = os.path.join(args.input_dir, args.domain_a_dir)
    domain_b_dir = os.path.join(args.input_dir, args.domain_b_dir)

    domain_a_files = find_image_files(domain_a_dir)
    domain_b_files = find_image_files(domain_b_dir)

    print(f"Found {len(domain_a_files)} files in domain A and {len(domain_b_files)} files in domain B")

    # Pair images
    if args.pair_mode == "filename":
        pairs = pair_files_by_name(domain_a_files, domain_b_files)
    elif args.pair_mode == "regex" and args.regex_pattern:
        # Extract keys using regex
        domain_a_matches = match_files_by_pattern(domain_a_files, args.regex_pattern)
        domain_b_matches = match_files_by_pattern(domain_b_files, args.regex_pattern)

        # Find common keys
        common_keys = set(domain_a_matches.keys()).intersection(set(domain_b_matches.keys()))

        # Create pairs
        pairs = [(domain_a_matches[k], domain_b_matches[k]) for k in common_keys]
    elif args.pair_mode == "order":
        # Pair by order (requires same number of files)
        min_len = min(len(domain_a_files), len(domain_b_files))
        pairs = list(zip(domain_a_files[:min_len], domain_b_files[:min_len]))
    else:
        raise ValueError(f"Invalid pair mode: {args.pair_mode}")

    print(f"Created {len(pairs)} image pairs")

    # Shuffle pairs
    random.shuffle(pairs)

    # Split into train, val, and test
    train_size = int(len(pairs) * split_ratio[0])
    val_size = int(len(pairs) * split_ratio[1])

    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:train_size + val_size]
    test_pairs = pairs[train_size + val_size:]

    print(f"Split into {len(train_pairs)} train, {len(val_pairs)} val, and {len(test_pairs)} test pairs")

    # Create output directories
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    test_dir = os.path.join(args.output_dir, "test")

    ensure_dir(train_dir)
    ensure_dir(val_dir)
    ensure_dir(test_dir)

    # Process and save paired images
    dataset_info = {
        "train": process_aligned_pairs(train_pairs, train_dir, args),
        "val": process_aligned_pairs(val_pairs, val_dir, args),
        "test": process_aligned_pairs(test_pairs, test_dir, args)
    }

    # Save dataset info
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump({
            "dataset_type": "aligned",
            "direction": args.direction,
            "pair_mode": args.pair_mode,
            "split_ratio": split_ratio,
            "img_size": args.img_size if not args.no_resize else None,
            "seed": args.seed,
            "num_pairs": {
                "train": len(train_pairs),
                "val": len(val_pairs),
                "test": len(test_pairs)
            },
            "dataset_info": dataset_info
        }, f, indent=2)

    print(f"Dataset preparation complete. Results saved to {args.output_dir}")
    return dataset_info


def process_aligned_pairs(pairs, output_dir, args):
    """
    Process and save aligned image pairs.

    Args:
        pairs: List of image pairs
        output_dir: Output directory
        args: Command line arguments

    Returns:
        List of processed pairs information
    """
    processed = []

    for i, (a_path, b_path) in enumerate(tqdm(pairs, desc=f"Processing {os.path.basename(output_dir)} pairs")):
        # Load images
        img_a = Image.open(a_path).convert('RGB')
        img_b = Image.open(b_path).convert('RGB')

        # Resize if needed
        if not args.no_resize:
            img_a = img_a.resize((args.img_size, args.img_size), Image.BICUBIC)
            img_b = img_b.resize((args.img_size, args.img_size), Image.BICUBIC)
        elif img_a.size != img_b.size:
            # Ensure same size if not resizing
            img_b = img_b.resize(img_a.size, Image.BICUBIC)

        # Create combined image (A|B)
        width, height = img_a.size
        combined = Image.new('RGB', (width * 2, height))
        combined.paste(img_a, (0, 0))
        combined.paste(img_b, (width, 0))

        # Save combined image
        output_path = os.path.join(output_dir, f"{i:04d}.png")
        combined.save(output_path)

        processed.append({
            "id": i,
            "a_path": a_path,
            "b_path": b_path,
            "combined_path": output_path
        })

    return processed


def prepare_unaligned_dataset(args):
    """
    Prepare an unaligned dataset (e.g., for CycleGAN).

    Args:
        args: Command line arguments

    Returns:
        Dictionary with dataset information
    """
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse split ratio
    split_ratio = tuple(map(float, args.split_ratio.split(',')))
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        print(f"Warning: Split ratio {split_ratio} does not sum to 1, normalizing")
        total = sum(split_ratio)
        split_ratio = tuple(r / total for r in split_ratio)

    # Find domain A and B images
    domain_a_dir = os.path.join(args.input_dir, args.domain_a_dir)
    domain_b_dir = os.path.join(args.input_dir, args.domain_b_dir)

    domain_a_files = find_image_files(domain_a_dir)
    domain_b_files = find_image_files(domain_b_dir)

    print(f"Found {len(domain_a_files)} files in domain A and {len(domain_b_files)} files in domain B")

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

    print(f"Split into {len(train_a_files)}/{len(train_b_files)} train, "
          f"{len(val_a_files)}/{len(val_b_files)} val, and "
          f"{len(test_a_files)}/{len(test_b_files)} test files")

    # Create output directories
    train_a_dir = os.path.join(args.output_dir, "trainA")
    train_b_dir = os.path.join(args.output_dir, "trainB")
    val_a_dir = os.path.join(args.output_dir, "valA")
    val_b_dir = os.path.join(args.output_dir, "valB")
    test_a_dir = os.path.join(args.output_dir, "testA")
    test_b_dir = os.path.join(args.output_dir, "testB")

    ensure_dir(train_a_dir)
    ensure_dir(train_b_dir)
    ensure_dir(val_a_dir)
    ensure_dir(val_b_dir)
    ensure_dir(test_a_dir)
    ensure_dir(test_b_dir)

    # Process and save images
    dataset_info = {
        "trainA": process_unaligned_files(train_a_files, train_a_dir, args),
        "trainB": process_unaligned_files(train_b_files, train_b_dir, args),
        "valA": process_unaligned_files(val_a_files, val_a_dir, args),
        "valB": process_unaligned_files(val_b_files, val_b_dir, args),
        "testA": process_unaligned_files(test_a_files, test_a_dir, args),
        "testB": process_unaligned_files(test_b_files, test_b_dir, args)
    }

    # Save dataset info
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump({
            "dataset_type": "unaligned",
            "direction": args.direction,
            "split_ratio": split_ratio,
            "img_size": args.img_size if not args.no_resize else None,
            "seed": args.seed,
            "num_files": {
                "trainA": len(train_a_files),
                "trainB": len(train_b_files),
                "valA": len(val_a_files),
                "valB": len(val_b_files),
                "testA": len(test_a_files),
                "testB": len(test_b_files)
            },
            "dataset_info": dataset_info
        }, f, indent=2)

    print(f"Dataset preparation complete. Results saved to {args.output_dir}")
    return dataset_info


def process_unaligned_files(files, output_dir, args):
    """
    Process and save unaligned image files.

    Args:
        files: List of image files
        output_dir: Output directory
        args: Command line arguments

    Returns:
        List of processed files information
    """
    processed = []

    for i, path in enumerate(tqdm(files, desc=f"Processing {os.path.basename(output_dir)} files")):
        # Load image
        img = Image.open(path).convert('RGB')

        # Resize if needed
        if not args.no_resize:
            img = img.resize((args.img_size, args.img_size), Image.BICUBIC)

        # Save image
        output_path = os.path.join(output_dir, f"{i:04d}.png")
        img.save(output_path)

        processed.append({
            "id": i,
            "original_path": path,
            "processed_path": output_path
        })

    return processed


def prepare_single_dataset(args):
    """
    Prepare a single-domain dataset.

    Args:
        args: Command line arguments

    Returns:
        Dictionary with dataset information
    """
    # Find images
    if args.domain_a_dir:
        input_dir = os.path.join(args.input_dir, args.domain_a_dir)
    else:
        input_dir = args.input_dir

    files = find_image_files(input_dir)
    print(f"Found {len(files)} files")

    # Create output directory
    ensure_dir(args.output_dir)

    # Process and save images
    dataset_info = process_unaligned_files(files, args.output_dir, args)

    # Save dataset info
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump({
            "dataset_type": "single",
            "img_size": args.img_size if not args.no_resize else None,
            "num_files": len(files),
            "dataset_info": dataset_info
        }, f, indent=2)

    print(f"Dataset preparation complete. Results saved to {args.output_dir}")
    return dataset_info


def main():
    args = parse_args()

    # Create output directory
    ensure_dir(args.output_dir)

    # Prepare dataset based on type
    if args.dataset_type == "aligned":
        prepare_aligned_dataset(args)
    elif args.dataset_type == "unaligned":
        prepare_unaligned_dataset(args)
    elif args.dataset_type == "single":
        prepare_single_dataset(args)
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")


if __name__ == "__main__":
    main()