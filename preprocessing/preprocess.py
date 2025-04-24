import argparse
import os
import sys
from PIL import Image
from tqdm import tqdm
import json
import numpy as np

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cache import cache_data, check_cache, get_cached_data


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


def resize_image(img, size, interpolation=Image.BICUBIC):
    """Resize an image to the specified size"""
    return img.resize(size, interpolation)


def crop_image(img, pos, size):
    """Crop an image at position pos to size"""
    return img.crop((pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]))


def random_crop(img, size):
    """Random crop an image to size"""
    w, h = img.size
    tw, th = size
    if w == tw and h == th:
        return img

    x1 = np.random.randint(0, w - tw)
    y1 = np.random.randint(0, h - th)
    return crop_image(img, (x1, y1), (tw, th))


def scale_width(img, target_width):
    """Scale image to target width while maintaining aspect ratio"""
    w, h = img.size
    target_height = int(target_width * h / w)
    return img.resize((target_width, target_height), Image.BICUBIC)


def preprocess_image(img, params):
    """
    Preprocess a single image based on parameters

    Args:
        img: PIL Image
        params: Dictionary of preprocessing parameters

    Returns:
        Preprocessed PIL Image
    """
    method = params['preprocess']
    load_size = params['load_size']
    crop_size = params['crop_size']

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
    if not params['no_flip'] and np.random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img


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
        return get_cached_data(cache_key)

    # Get image paths
    print(f"Preprocessing aligned dataset from {input_dir}")
    image_paths = []
    for root, _, fnames in sorted(os.walk(input_dir)):
        for fname in fnames:
            if any(fname.endswith(ext) for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']):
                path = os.path.join(root, fname)
                image_paths.append(path)

    # Process images
    processed_images = []
    for i, path in enumerate(tqdm(image_paths, desc="Processing images")):
        # Open image
        img = Image.open(path).convert('RGB')

        # Split A and B images for alignment
        w, h = img.size
        w2 = int(w / 2)

        # Get images based on direction
        if opt.direction == 'AtoB':
            img_A = img.crop((0, 0, w2, h))
            img_B = img.crop((w2, 0, w, h))
        else:
            img_A = img.crop((w2, 0, w, h))
            img_B = img.crop((0, 0, w2, h))

        # Preprocess images
        img_A = preprocess_image(img_A, transform_params)
        img_B = preprocess_image(img_B, transform_params)

        # Combine A and B
        processed_img = Image.new('RGB', (img_A.width + img_B.width, img_A.height))
        processed_img.paste(img_A, (0, 0))
        processed_img.paste(img_B, (img_A.width, 0))

        # Save processed image
        output_path = os.path.join(output_dir, f"{i:04d}.png")
        processed_img.save(output_path)

        processed_images.append({
            'original_path': path,
            'processed_path': output_path,
            'A_size': img_A.size,
            'B_size': img_B.size
        })

    # Cache the preprocessed data
    cache_data(cache_key, {
        'transform_params': transform_params,
        'output_dir': output_dir,
        'processed_images': processed_images
    })

    print(f"Preprocessing complete. Processed {len(image_paths)} images to {output_dir}.")
    return {
        'transform_params': transform_params,
        'output_dir': output_dir,
        'processed_images': processed_images
    }


def preprocess_unaligned_dataset(opt):
    """Preprocess unaligned dataset (like CycleGAN)"""
    # Define directories
    input_dir = opt.input_dir
    output_dir = opt.output_dir
    phase = opt.phase

    # Look for A and B directories
    dir_A = os.path.join(input_dir, f'{phase}A')
    dir_B = os.path.join(input_dir, f'{phase}B')

    # Check if directories exist, if not try alternate structure
    if not os.path.exists(dir_A) or not os.path.exists(dir_B):
        dir_A = os.path.join(input_dir, 'A')
        dir_B = os.path.join(input_dir, 'B')

        if not os.path.exists(dir_A) or not os.path.exists(dir_B):
            raise ValueError(f"Could not find directories for domains A and B in {input_dir}")

    # Create output directories
    output_dir_A = os.path.join(output_dir, f'{phase}A')
    output_dir_B = os.path.join(output_dir, f'{phase}B')
    os.makedirs(output_dir_A, exist_ok=True)
    os.makedirs(output_dir_B, exist_ok=True)

    # Define transformations
    transform_params = {
        'load_size': opt.load_size,
        'crop_size': opt.crop_size,
        'preprocess': opt.preprocess,
        'no_flip': opt.no_flip,
        'direction': opt.direction
    }

    # Check if preprocessed data exists in cache
    cache_key = f"unaligned_{input_dir}_{phase}_{opt.preprocess}_{opt.load_size}_{opt.crop_size}"
    if not opt.force and check_cache(cache_key):
        print(f"Using cached preprocessed data for {cache_key}")
        return get_cached_data(cache_key)

    # Get image paths for domain A
    image_paths_A = []
    for root, _, fnames in sorted(os.walk(dir_A)):
        for fname in fnames:
            if any(fname.endswith(ext) for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']):
                path = os.path.join(root, fname)
                image_paths_A.append(path)

    # Get image paths for domain B
    image_paths_B = []
    for root, _, fnames in sorted(os.walk(dir_B)):
        for fname in fnames:
            if any(fname.endswith(ext) for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']):
                path = os.path.join(root, fname)
                image_paths_B.append(path)

    print(f"Found {len(image_paths_A)} images in domain A and {len(image_paths_B)} images in domain B")

    # Process domain A images
    processed_images_A = []
    for i, path in enumerate(tqdm(image_paths_A, desc="Processing domain A")):
        # Open image
        img = Image.open(path).convert('RGB')

        # Preprocess image
        processed_img = preprocess_image(img, transform_params)

        # Save processed image
        output_path = os.path.join(output_dir_A, f"{i:04d}.png")
        processed_img.save(output_path)

        processed_images_A.append({
            'original_path': path,
            'processed_path': output_path,
            'size': processed_img.size
        })

    # Process domain B images
    processed_images_B = []
    for i, path in enumerate(tqdm(image_paths_B, desc="Processing domain B")):
        # Open image
        img = Image.open(path).convert('RGB')

        # Preprocess image
        processed_img = preprocess_image(img, transform_params)

        # Save processed image
        output_path = os.path.join(output_dir_B, f"{i:04d}.png")
        processed_img.save(output_path)

        processed_images_B.append({
            'original_path': path,
            'processed_path': output_path,
            'size': processed_img.size
        })

    # Cache the preprocessed data
    result = {
        'transform_params': transform_params,
        'output_dir_A': output_dir_A,
        'output_dir_B': output_dir_B,
        'processed_images_A': processed_images_A,
        'processed_images_B': processed_images_B
    }

    cache_data(cache_key, result)

    print(
        f"Preprocessing complete. Processed {len(image_paths_A)} domain A images and {len(image_paths_B)} domain B images.")
    return result


def preprocess_single_dataset(opt):
    """Preprocess single dataset for testing/inference"""
    # Define directories
    input_dir = opt.input_dir
    output_dir = opt.output_dir
    phase = opt.phase

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Define transformations - for test phase, usually only resize, no crop or flip
    transform_params = {
        'load_size': opt.load_size,
        'crop_size': opt.crop_size,
        'preprocess': 'scale_width' if opt.preprocess == 'scale_width_and_crop' else opt.preprocess,
        'no_flip': True,  # Don't flip test images
        'direction': opt.direction
    }

    # Get image paths
    print(f"Preprocessing single dataset from {input_dir}")
    image_paths = []
    for root, _, fnames in sorted(os.walk(input_dir)):
        for fname in fnames:
            if any(fname.endswith(ext) for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']):
                path = os.path.join(root, fname)
                image_paths.append(path)

    # Process images
    processed_images = []
    for i, path in enumerate(tqdm(image_paths, desc="Processing images")):
        # Open image
        img = Image.open(path).convert('RGB')

        # Preprocess image
        processed_img = preprocess_image(img, transform_params)

        # Save processed image
        output_path = os.path.join(output_dir, f"{i:04d}.png")
        processed_img.save(output_path)

        processed_images.append({
            'original_path': path,
            'processed_path': output_path,
            'size': processed_img.size
        })

    print(f"Preprocessing complete. Processed {len(image_paths)} images to {output_dir}.")
    return {
        'transform_params': transform_params,
        'output_dir': output_dir,
        'processed_images': processed_images
    }


def main():
    """Main function"""
    # Parse arguments
    opt = parse_args()

    # Load config
    config = load_config(opt.config)

    # Override config with command line arguments
    for key, value in config.items():
        if not hasattr(opt, key) or getattr(opt, key) is None:
            setattr(opt, key, value)

    # Set input and output directories
    if not hasattr(opt, 'input_dir') or opt.input_dir is None:
        opt.input_dir = config.get('dataroot', 'datasets')
    if not hasattr(opt, 'output_dir') or opt.output_dir is None:
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