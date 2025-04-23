import argparse
import os
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import json

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cache import cache_data, check_cache


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Preprocess data for CycleGAN/pix2pix')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input_dir', type=str, help='Input directory (overrides config)')
    parser.add_argument('--output_dir', type=str, help='Output directory (overrides config)')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--load_size', type=int, default=286, help='Scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='Then crop to this size')
    parser.add_argument('--no_flip', action='store_true', help='If specified, do not flip')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                        help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--dataset_mode', type=str, default='aligned',
                        help='chooses how datasets are loaded [aligned | unaligned | single]')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--force', action='store_true', help='Force preprocessing even if cache exists')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def preprocess_aligned_dataset(opt):
    """Preprocess aligned dataset (like pix2pix)"""
    # Define directories
    input_dir = opt.input_dir
    output_dir = opt.output_dir
    phase = opt.phase

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Define transformations
    transform_params = {
        'load_size': opt.load_size,
        'crop_size': opt.crop_size,
        'preprocess': opt.preprocess,
        'no_flip': opt.no_flip,
        'direction': opt.direction
    }

    # Check if preprocessed data exists in cache
    cache_key = f"aligned_{input_dir}_{phase}_{opt.preprocess}_{opt.load_size}_{opt.crop_size}"
    if not opt.force and check_cache(cache_key):
        print(f"Using cached preprocessed data for {cache_key}")
        return

    # Get image paths
    print(f"Preprocessing aligned dataset from {input_dir}")
    image_paths = []
    for root, _, fnames in sorted(os.walk(input_dir)):
        for fname in fnames:
            if any(fname.endswith(ext) for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']):
                path = os.path.join(root, fname)
                image_paths.append(path)

    # Process images
    for i, path in enumerate(tqdm(image_paths)):
        # Open image
        img = Image.open(path).convert('RGB')

        # Preprocess image
        # ... (Implementation of preprocessing using transform_params)

        # Save processed image
        # ... (Save to output_dir)

    # Cache the preprocessed data
    cache_data(cache_key, {'transform_params': transform_params, 'output_dir': output_dir})
    print(f"Preprocessing complete. Processed {len(image_paths)} images.")


def preprocess_unaligned_dataset(opt):
    """Preprocess unaligned dataset (like CycleGAN)"""
    # Similar implementation as above but for unaligned datasets
    pass


def preprocess_single_dataset(opt):
    """Preprocess single dataset"""
    # Similar implementation as above but for single datasets
    pass


def main():
    """Main function"""
    # Parse arguments
    opt = parse_args()

    # Load config
    config = load_config(opt.config)

    # Override config with command line arguments
    if opt.input_dir:
        config['dataroot'] = opt.input_dir
    if opt.output_dir:
        config['preprocessed_dataroot'] = opt.output_dir

    # Update opt with config values
    for key, value in config.items():
        if not hasattr(opt, key) or getattr(opt, key) is None:
            setattr(opt, key, value)

    # Set input and output directories
    opt.input_dir = config.get('dataroot', 'datasets')
    opt.output_dir = config.get('preprocessed_dataroot', 'datasets_preprocessed')

    # Call appropriate preprocessing function based on dataset mode
    if opt.dataset_mode == 'aligned':
        preprocess_aligned_dataset(opt)
    elif opt.dataset_mode == 'unaligned':
        preprocess_unaligned_dataset(opt)
    elif opt.dataset_mode == 'single':
        preprocess_single_dataset(opt)
    else:
        raise ValueError(f"Unknown dataset mode: {opt.dataset_mode}")


if __name__ == '__main__':
    main()