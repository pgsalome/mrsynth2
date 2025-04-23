"""
MRI slice extraction methods

This module provides methods for extracting 2D slices from 3D MRI volumes
for use in 2D image-to-image translation models.
"""

import os
import numpy as np
import cv2
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import SimpleITK as sitk

    HAS_SITK = True
except ImportError:
    HAS_SITK = False

# Set up logging
import logging

logger = logging.getLogger(__name__)


def load_volume(file_path: str) -> Dict[str, Any]:
    """
    Load a 3D MRI volume

    Args:
        file_path: Path to the volume file

    Returns:
        Dictionary containing volume data and metadata
    """
    if HAS_NIBABEL:
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            affine = img.affine
            header = img.header
            return {'data': data, 'affine': affine, 'header': header, 'file_format': 'nibabel'}
        except Exception as e:
            logger.warning(f"Failed to load with nibabel: {e}")

    if HAS_SITK:
        try:
            img = sitk.ReadImage(file_path)
            data = sitk.GetArrayFromImage(img)
            # SimpleITK uses a different axis ordering (z, y, x) vs (x, y, z)
            data = np.transpose(data, list(range(data.ndim))[::-1])
            return {'data': data, 'sitk_img': img, 'file_format': 'sitk'}
        except Exception as e:
            logger.warning(f"Failed to load with SimpleITK: {e}")

    raise ImportError("No NIfTI loader available")


def extract_slice(volume_data: np.ndarray, slice_idx: int, axis: int = 0) -> np.ndarray:
    """
    Extract a 2D slice from a 3D volume along a specified axis

    Args:
        volume_data: 3D volume data
        slice_idx: Slice index to extract
        axis: Axis along which to extract the slice (0, 1, or 2)

    Returns:
        2D slice as a numpy array
    """
    if axis == 0:
        return volume_data[slice_idx, :, :]
    elif axis == 1:
        return volume_data[:, slice_idx, :]
    elif axis == 2:
        return volume_data[:, :, slice_idx]
    else:
        raise ValueError(f"Invalid axis: {axis}")


def has_sufficient_content(slice_data: np.ndarray, mask_data: Optional[np.ndarray] = None,
                           threshold: float = 0.05, intensity_threshold: float = 0.1) -> bool:
    """
    Check if a slice has sufficient content to be included

    Args:
        slice_data: 2D slice data
        mask_data: Optional 2D mask data
        threshold: Threshold for content (fraction of non-zero voxels)
        intensity_threshold: Threshold for intensity (used if no mask is provided)

    Returns:
        True if slice has sufficient content, False otherwise
    """
    if mask_data is not None:
        # Use mask to determine content
        content_ratio = np.sum(mask_data > 0) / mask_data.size
    else:
        # Use image intensity to determine content
        # Normalize the slice
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)

        if max_val == min_val:
            return False

        normalized = (slice_data - min_val) / (max_val - min_val)

        # Threshold to create a foreground mask
        foreground = normalized > intensity_threshold

        # Calculate content ratio
        content_ratio = np.sum(foreground) / foreground.size

    return content_ratio >= threshold


def normalize_slice(slice_data: np.ndarray, method: str = 'min_max', mask: Optional[np.ndarray] = None,
                    params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Normalize a 2D slice

    Args:
        slice_data: 2D slice data
        method: Normalization method ('min_max', 'z_score', 'percentile')
        mask: Optional mask to apply before calculating statistics
        params: Additional parameters for normalization

    Returns:
        Normalized 2D slice
    """
    if params is None:
        params = {}

    if method == 'min_max':
        min_val = np.min(slice_data) if mask is None else np.min(slice_data[mask > 0])
        max_val = np.max(slice_data) if mask is None else np.max(slice_data[mask > 0])

        if max_val == min_val:
            return np.zeros_like(slice_data)

        new_min = params.get('new_min', 0.0)
        new_max = params.get('new_max', 1.0)
        return (slice_data - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

    elif method == 'z_score':
        mean = np.mean(slice_data) if mask is None else np.mean(slice_data[mask > 0])
        std = np.std(slice_data) if mask is None else np.std(slice_data[mask > 0])

        if std == 0:
            return np.zeros_like(slice_data)

        return (slice_data - mean) / std

    elif method == 'percentile':
        p_low = params.get('p_low', 1)
        p_high = params.get('p_high', 99)

        p_low_val = np.percentile(slice_data, p_low) if mask is None else np.percentile(slice_data[mask > 0], p_low)
        p_high_val = np.percentile(slice_data, p_high) if mask is None else np.percentile(slice_data[mask > 0], p_high)

        if p_high_val == p_low_val:
            return np.zeros_like(slice_data)

        # Clip data to percentile range and normalize
        clipped = np.clip(slice_data, p_low_val, p_high_val)

        new_min = params.get('new_min', 0.0)
        new_max = params.get('new_max', 1.0)
        return (clipped - p_low_val) / (p_high_val - p_low_val) * (new_max - new_min) + new_min

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_slice(slice_data: np.ndarray, target_size: Tuple[int, int],
                 interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """
    Resize a 2D slice to the target size

    Args:
        slice_data: 2D slice data
        target_size: Target size as (height, width)
        interpolation: Interpolation method

    Returns:
        Resized 2D slice
    """
    # Convert to float32 to avoid precision loss
    slice_data = slice_data.astype(np.float32)

    # Normalize to [0, 1] if needed
    min_val = np.min(slice_data)
    max_val = np.max(slice_data)

    if min_val < 0 or max_val > 1:
        slice_data = (slice_data - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(slice_data)

    # Scale to [0, 255] for cv2
    slice_data = (slice_data * 255).astype(np.uint8)

    # Resize using cv2
    resized = cv2.resize(slice_data, (target_size[1], target_size[0]), interpolation=interpolation)

    # Scale back to [0, 1]
    return resized.astype(np.float32) / 255.0


def save_slice(slice_data: np.ndarray, output_path: str, normalize: bool = True) -> str:
    """
    Save a 2D slice as an image file

    Args:
        slice_data: 2D slice data
        output_path: Path to save the slice
        normalize: Whether to normalize the slice before saving

    Returns:
        Path to the saved slice
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Normalize if requested
    if normalize:
        slice_data = normalize_slice(slice_data)

    # Scale to [0, 255] for saving
    slice_data = (slice_data * 255).astype(np.uint8)

    # Save the image
    cv2.imwrite(output_path, slice_data)

    return output_path


def generate_slices(volume_path: str, output_dir: str, axis: int = 2, slice_range: Optional[Tuple[int, int]] = None,
                    slice_step: int = 1, target_size: Optional[Tuple[int, int]] = None,
                    mask_path: Optional[str] = None, content_threshold: float = 0.05,
                    normalization: str = 'min_max', normalization_params: Optional[Dict[str, Any]] = None,
                    prefix: str = '') -> List[Dict[str, Any]]:
    """
    Generate 2D slices from a 3D volume

    Args:
        volume_path: Path to the volume file
        output_dir: Directory to save slices
        axis: Axis along which to extract slices (0, 1, or 2)
        slice_range: Range of slices to extract (start, end)
        slice_step: Step size for slice extraction
        target_size: Target size for slices (height, width)
        mask_path: Path to mask volume
        content_threshold: Threshold for slice content
        normalization: Normalization method
        normalization_params: Additional parameters for normalization
        prefix: Prefix for slice filenames

    Returns:
        List of dictionaries with information about extracted slices
    """
    # Load volume
    volume = load_volume(volume_path)
    volume_data = volume['data']

    # Load mask if provided
    mask_data = None
    if mask_path and os.path.exists(mask_path):
        mask = load_volume(mask_path)
        mask_data = mask['data']

        # Check if mask has the same shape as volume
        if mask_data.shape != volume_data.shape:
            logger.warning(f"Mask shape {mask_data.shape} does not match volume shape {volume_data.shape}")
            mask_data = None

    # Determine slice range if not provided
    if slice_range is None:
        if axis == 0:
            slice_range = (0, volume_data.shape[0])
        elif axis == 1:
            slice_range = (0, volume_data.shape[1])
        elif axis == 2:
            slice_range = (0, volume_data.shape[2])

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate slices
    slices_info = []

    for slice_idx in range(slice_range[0], slice_range[1], slice_step):
        # Extract the slice
        slice_data = extract_slice(volume_data, slice_idx, axis)

        # Extract the mask slice if available
        mask_slice = None
        if mask_data is not None:
            mask_slice = extract_slice(mask_data, slice_idx, axis)

        # Check if slice has sufficient content
        if not has_sufficient_content(slice_data, mask_slice, content_threshold):
            continue

        # Resize the slice if needed
        if target_size and (slice_data.shape[0] != target_size[0] or slice_data.shape[1] != target_size[1]):
            slice_data = resize_slice(slice_data, target_size)
            if mask_slice is not None:
                mask_slice = resize_slice(mask_slice, target_size)

        # Create output path
        volume_name = os.path.splitext(os.path.basename(volume_path))[0].split('.')[0]  # Remove all extensions
        output_path = os.path.join(
            output_dir,
            f"{prefix}{volume_name}_axis{axis}_slice{slice_idx:03d}.png"
        )

        # Save the slice
        save_slice(slice_data, output_path, normalize=(normalization != 'none'))

        # Save mask slice if available
        mask_output_path = None
        if mask_slice is not None:
            mask_output