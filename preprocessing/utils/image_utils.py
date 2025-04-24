"""Utilities for image manipulation in preprocessing pipelines."""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, Any, Union, List
import os
from pathlib import Path


def load_image(file_path: str, as_grayscale: bool = False) -> np.ndarray:
    """
    Load an image from file.

    Args:
        file_path: Path to image file
        as_grayscale: Whether to load as grayscale

    Returns:
        Loaded image as numpy array
    """
    if as_grayscale:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    else:
        # Load with OpenCV (BGR) and convert to RGB
        img = cv2.imread(file_path)
        if img is not None and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        # Try with PIL if OpenCV fails
        try:
            pil_img = Image.open(file_path)
            if as_grayscale:
                pil_img = pil_img.convert('L')
            img = np.array(pil_img)
        except Exception as e:
            raise ValueError(f"Could not load image {file_path}: {e}")

    return img


def save_image(img: np.ndarray, file_path: str) -> None:
    """
    Save an image to file.

    Args:
        img: Image as numpy array
        file_path: Path to save image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Determine if image is grayscale or color
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        # Grayscale image
        cv2.imwrite(file_path, img)
    else:
        # Color image (convert from RGB to BGR for OpenCV)
        cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def resize_image(img: np.ndarray, size: Tuple[int, int],
                 interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """
    Resize an image.

    Args:
        img: Image as numpy array
        size: Target size as (width, height)
        interpolation: Interpolation method

    Returns:
        Resized image
    """
    # Note: cv2.resize takes (width, height) while img.shape is (height, width, channels)
    return cv2.resize(img, size, interpolation=interpolation)


def normalize_image(img: np.ndarray, method: str = 'min_max',
                    new_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Normalize image values.

    Args:
        img: Image as numpy array
        method: Normalization method ('min_max', 'z_score', or 'percentile')
        new_range: Target range for normalized values

    Returns:
        Normalized image
    """
    # Convert to float for normalization
    img_float = img.astype(np.float32)

    if method == 'min_max':
        min_val = np.min(img_float)
        max_val = np.max(img_float)

        if max_val == min_val:
            # Handle constant value image
            return np.zeros_like(img_float)

        # Normalize to [0, 1]
        img_norm = (img_float - min_val) / (max_val - min_val)

    elif method == 'z_score':
        mean = np.mean(img_float)
        std = np.std(img_float)

        if std == 0:
            # Handle constant value image
            return np.zeros_like(img_float)

        # Normalize to mean=0, std=1
        img_norm = (img_float - mean) / std

    elif method == 'percentile':
        p1 = np.percentile(img_float, 1)
        p99 = np.percentile(img_float, 99)

        if p99 == p1:
            # Handle constant value image
            return np.zeros_like(img_float)

        # Clip and normalize to [0, 1]
        img_clip = np.clip(img_float, p1, p99)
        img_norm = (img_clip - p1) / (p99 - p1)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Scale to new range
    if new_range != (0, 1):
        img_norm = img_norm * (new_range[1] - new_range[0]) + new_range[0]

    return img_norm


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.

    Args:
        img: Image as numpy array

    Returns:
        Grayscale image
    """
    if img.ndim == 2:
        # Already grayscale
        return img
    elif img.ndim == 3:
        if img.shape[2] == 1:
            # Single channel image
            return img[:, :, 0]
        else:
            # Multi-channel image
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unexpected image dimensions: {img.shape}")


def apply_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """
    Apply windowing to a medical image (e.g., CT, MRI).

    Args:
        img: Image as numpy array
        window_center: Window center (level)
        window_width: Window width

    Returns:
        Windowed image with values in [0, 1]
    """
    # Calculate min and max values
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2

    # Clip values to window range
    img_windowed = np.clip(img, min_val, max_val)

    # Normalize to [0, 1]
    img_windowed = (img_windowed - min_val) / (max_val - min_val)

    return img_windowed


def apply_mask(img: np.ndarray, mask: np.ndarray,
               background_value: Union[float, Tuple[float, ...]] = 0) -> np.ndarray:
    """
    Apply a binary mask to an image.

    Args:
        img: Image as numpy array
        mask: Binary mask (0 for background, >0 for foreground)
        background_value: Value to use for masked-out regions

    Returns:
        Masked image
    """
    # Ensure mask is binary
    binary_mask = mask > 0

    # Expand mask dimensions if needed
    if img.ndim == 3 and binary_mask.ndim == 2:
        binary_mask = np.expand_dims(binary_mask, axis=2)

    # Create output image with background value
    if isinstance(background_value, (int, float)):
        # Single value for all channels
        result = np.full_like(img, background_value)
    else:
        # Tuple of values for each channel
        result = np.zeros_like(img)
        for i, val in enumerate(background_value):
            if img.ndim == 3:
                result[:, :, i] = val
            else:
                result[:] = val

    # Apply mask
    np.copyto(result, img, where=binary_mask)

    return result


def create_composite_image(images: List[np.ndarray], layout: Tuple[int, int] = None,
                           spacing: int = 5) -> np.ndarray:
    """
    Create a composite image from a list of images.

    Args:
        images: List of images as numpy arrays
        layout: Layout as (rows, cols), if None, calculated automatically
        spacing: Spacing between images in pixels

    Returns:
        Composite image
    """
    n_images = len(images)

    if n_images == 0:
        return np.zeros((100, 100), dtype=np.uint8)

    # Determine layout if not provided
    if layout is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        layout = (rows, cols)
    else:
        rows, cols = layout

    # Ensure all images have the same shape and type
    first_img = images[0]
    h, w = first_img.shape[:2]
    is_color = first_img.ndim == 3

    # Create empty composite image
    if is_color:
        composite = np.zeros(((h + spacing) * rows - spacing,
                              (w + spacing) * cols - spacing,
                              first_img.shape[2]), dtype=first_img.dtype)
    else:
        composite = np.zeros(((h + spacing) * rows - spacing,
                              (w + spacing) * cols - spacing),
                             dtype=first_img.dtype)

    # Place images in grid
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break

        i = idx // cols
        j = idx % cols

        # Handle images with different dimensions
        if img.shape[:2] != (h, w):
            img = resize_image(img, (w, h))

        # Handle color/grayscale mismatch
        if is_color and img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif not is_color and img.ndim == 3:
            img = convert_to_grayscale(img)

        # Place image in grid
        y_start = i * (h + spacing)
        y_end = y_start + h
        x_start = j * (w + spacing)
        x_end = x_start + w

        if is_color:
            composite[y_start:y_end, x_start:x_end, :] = img
        else:
            composite[y_start:y_end, x_start:x_end] = img

    return composite