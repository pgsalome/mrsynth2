#!/usr/bin/env python
"""
Create 2D slices from 3D MRI volumes

This script extracts 2D slices from 3D MRI volumes for use in 2D image
translation models. It can extract slices along specific axes, filter
out slices with insufficient content, and create paired or unpaired datasets.
"""

import os
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import cv2

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
        raise ImportError("Could not find nibabel or SimpleITK. Please install at least one.")

# Try to import our own slice extraction module
try:
    from src.preprocessing.slice_extraction import extract_slices as slice_module

    HAS_SLICE_MODULE = True
except ImportError:
    HAS_SLICE_MODULE = False
    logging.warning("Custom slice extraction module not available.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_volume(file_path):
    """
    Load a 3D MRI volume

    Args:
        file_path: Path to 3D volume file

    Returns:
        Dictionary containing volume data, affine matrix, and header
    """
    if HAS_NIBABEL:
        img = nib.load(file_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        return {'data': data, 'affine': affine, 'header': header, 'file_format': 'nibabel'}
    elif HAS_SITK:
        img = sitk.ReadImage(file_path)
        data = sitk.GetArrayFromImage(img)
        # SimpleITK uses a different axis ordering (z, y, x) vs (x, y, z)
        data = np.transpose(data, list(range(data.ndim))[::-1])
        return {'data': data, 'sitk_img': img, 'file_format': 'sitk'}
    else:
        raise ImportError("No NIfTI loader available")


def extract_slice(volume_data, slice_idx, axis=0):
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


def has_sufficient_content(slice_data, mask_data=None, threshold=0.05):
    """
    Check if a slice has sufficient content to be included

    Args:
        slice_data: 2D slice data
        mask_data: Optional 2D mask data
        threshold: Threshold for content (fraction of non-zero voxels)

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
        foreground = normalized > 0.1

        # Calculate content ratio
        content_ratio = np.sum(foreground) / foreground.size

    return content_ratio >= threshold


def normalize_slice(slice_data, method='min_max'):
    """
    Normalize a 2D slice

    Args:
        slice_data: 2D slice data
        method: Normalization method ('min_max', 'z_score', 'percentile')

    Returns:
        Normalized 2D slice
    """
    if method == 'min_max':
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)

        if max_val == min_val:
            return np.zeros_like(slice_data)

        return (slice_data - min_val) / (max_val - min_val)

    elif method == 'z_score':
        mean = np.mean(slice_data)
        std = np.std(slice_data)

        if std == 0:
            return np.zeros_like(slice_data)

        return (slice_data - mean) / std

    elif method == 'percentile':
        p1 = np.percentile(slice_data, 1)
        p99 = np.percentile(slice_data, 99)

        if p99 == p1:
            return np.zeros_like(slice_data)

        normalized = (slice_data - p1) / (p99 - p1)
        return np.clip(normalized, 0, 1)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_slice(slice_data, target_size):
    """
    Resize a 2D slice to the target size

    Args:
        slice_data: 2D slice data
        target_size: Target size as (height, width)

    Returns:
        Resized 2D slice
    """
    # Convert to float32 to avoid precision loss
    slice_data = slice_data.astype(np.float32)

    # Scale to [0, 255] for cv2
    if slice_data.min() < 0 or slice_data.max() > 1:
        slice_data = normalize_slice(slice_data, 'min_max')

    slice_data = (slice_data * 255).astype(np.uint8)

    # Resize using cv2
    resized = cv2.resize(slice_data, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)

    # Scale back to [0, 1]
    return resized.astype(np.float32) / 255.0


def save_slice(slice_data, output_path, normalize=True):
    """
    Save a 2D slice as an image file

    Args:
        slice_data: 2D slice data
        output_path: Path to save the slice
        normalize: Whether to normalize the slice before saving
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


def process_volume(volume_info):
    """Process a single volume to extract slices"""
    try:
        volume_path = volume_info['volume_path']
        output_dir = volume_info['output_dir']
        subject_id = volume_info['subject_id']
        scan_type = volume_info['scan_type']
        axis = volume_info.get('axis', 2)
        target_size = volume_info.get('target_size', (256, 256))
        content_threshold = volume_info.get('content_threshold', 0.05)
        mask_path = volume_info.get('mask_path')
        slice_range = volume_info.get('slice_range')
        slice_step = volume_info.get('slice_step', 1)
        normalization = volume_info.get('normalization', 'min_max')
        paired_with = volume_info.get('paired_with')
        paired_output_dir = volume_info.get('paired_output_dir')

        # Load the volume
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

        # Extract and save slices
        valid_slices = []
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

            # Save the slice
            output_path = os.path.join(
                output_dir,
                f"{subject_id}_{scan_type}_axis{axis}_slice{slice_idx:03d}.png"
            )

            save_slice(slice_data, output_path, normalize=(normalization != 'none'))

            # If paired with another scan, create paired slice
            if paired_with and paired_output_dir:
                # Create paired output directory
                os.makedirs(paired_output_dir, exist_ok=True)

                # Get the paired slice
                paired_slice_data = extract_slice(paired_with['data'], slice_idx, axis)

                # Resize if needed
                if target_size and (
                        paired_slice_data.shape[0] != target_size[0] or paired_slice_data.shape[1] != target_size[1]):
                    paired_slice_data = resize_slice(paired_slice_data, target_size)

                # For paired data, save the slices side by side for pix2pix
                if volume_info.get('create_pix2pix', False):
                    # Create combined image (A|B)
                    combined = np.zeros((target_size[0], target_size[1] * 2))
                    combined[:, :target_size[1]] = slice_data
                    combined[:, target_size[1]:] = paired_slice_data

                    # Save combined image
                    paired_output_path = os.path.join(
                        paired_output_dir,
                        f"{subject_id}_paired_axis{axis}_slice{slice_idx:03d}.png"
                    )

                    save_slice(combined, paired_output_path, normalize=False)
                else:
                    # Save the paired slice separately
                    paired_output_path = os.path.join(
                        paired_output_dir,
                        f"{subject_id}_{paired_with['scan_type']}_axis{axis}_slice{slice_idx:03d}.png"
                    )

                    save_slice(paired_slice_data, paired_output_path, normalize=(normalization != 'none'))

            valid_slices.append({
                'slice_idx': slice_idx,
                'output_path': output_path
            })

        return {
            'subject_id': subject_id,
            'scan_type': scan_type,
            'volume_path': volume_path,
            'valid_slices': valid_slices
        }

    except Exception as e:
        logger.error(f"Error processing {volume_info['volume_path']}: {e}")
        return None


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Create 2D slices from 3D MRI volumes")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing 3D volumes")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for 2D slices")
    parser.add_argument("--volumes_list", type=str,
                        help="JSON file with list of volumes to process (optional)")
    parser.add_argument("--pair_scans", action="store_true",
                        help="Create paired data from different scan types of the same subject")
    parser.add_argument("--create_pix2pix", action="store_true",
                        help="Create combined images for pix2pix (A|B)")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                        help="Axis along which to extract slices (0=sagittal, 1=coronal, 2=axial)")
    parser.add_argument("--target_size", type=str, default="256,256",
                        help="Target size for slices as height,width")
    parser.add_argument("--content_threshold", type=float, default=0.05,
                        help="Threshold for slice content (fraction of non-zero voxels)")
    parser.add_argument("--slice_step", type=int, default=1,
                        help="Step size for extracting slices")
    parser.add_argument("--normalization", type=str, default="min_max",
                        choices=["min_max", "z_score", "percentile", "none"],
                        help="Normalization method for slices")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of volumes to process
    if args.volumes_list:
        # Load from JSON file
        with open(args.volumes_list, 'r') as f:
            volumes_info = json.load(f)
    else:
        # Auto-find NIfTI files in input directory
        volumes_info = []

        # Get all NIfTI files in the input directory
        nifti_files = []
        for root, _, files in os.walk(args.input_dir):
            for f in files:
                if f.endswith(('.nii', '.nii.gz')):
                    nifti_files.append(os.path.join(root, f))

        # Try to infer subject IDs and scan types from filenames
        volumes_by_subject = {}

        for file_path in nifti_files:
            filename = os.path.basename(file_path)
            stem = Path(filename).stem

            # Try to extract subject ID and scan type
            # Common patterns are: subject_scan.nii.gz, subject-scan.nii.gz, etc.
            parts = stem.replace('-', '_').split('_')

            if len(parts) >= 2:
                subject_id = parts[0]
                scan_type = parts[1]
            else:
                # If no clear pattern, use the whole filename as subject and 'unknown' as scan
                subject_id = stem
                scan_type = 'unknown'

            # Add to volumes by subject
            if subject_id not in volumes_by_subject:
                volumes_by_subject[subject_id] = []

            volumes_by_subject[subject_id].append({
                'volume_path': file_path,
                'scan_type': scan_type
            })

        # Create volume info objects
        for subject_id, volumes in volumes_by_subject.items():
            for volume in volumes:
                # Check for mask file
                mask_path = None
                volume_dir = os.path.dirname(volume['volume_path'])
                volume_name = os.path.basename(volume['volume_path'])
                mask_name = Path(volume_name).stem + '_mask.nii.gz'
                potential_mask_path = os.path.join(volume_dir, mask_name)

                if os.path.exists(potential_mask_path):
                    mask_path = potential_mask_path

                # Create volume info
                volumes_info.append({
                    'volume_path': volume['volume_path'],
                    'output_dir': os.path.join(args.output_dir, subject_id, volume['scan_type']),
                    'subject_id': subject_id,
                    'scan_type': volume['scan_type'],
                    'axis': args.axis,
                    'target_size': target_size,
                    'content_threshold': args.content_threshold,
                    'mask_path': mask_path,
                    'slice_step': args.slice_step,
                    'normalization': args.normalization
                })

    # Update volume info for paired data if requested
    if args.pair_scans:
        # Group volumes by subject
        volumes_by_subject = {}
        for info in volumes_info:
            subject_id = info['subject_id']
            if subject_id not in volumes_by_subject:
                volumes_by_subject[subject_id] = []
            volumes_by_subject[subject_id].append(info)

        # Create paired data for subjects with multiple scan types
        paired_volumes_info = []

        for subject_id, volumes in volumes_by_subject.items():
            if len(volumes) >= 2:
                # Create pairs
                for i, vol1 in enumerate(volumes):
                    for j, vol2 in enumerate(volumes):
                        if i != j:
                            # Clone volume info
                            paired_info = vol1.copy()

                            # Load the paired volume
                            paired_volume = load_volume(vol2['volume_path'])

                            # Add paired info
                            paired_info['paired_with'] = {
                                'data': paired_volume['data'],
                                'scan_type': vol2['scan_type']
                            }

                            # Create output directory for paired data
                            if args.create_pix2pix:
                                # For pix2pix, create a combined A|B image
                                paired_info['paired_output_dir'] = os.path.join(
                                    args.output_dir,
                                    'paired',
                                    f"{subject_id}_{vol1['scan_type']}_{vol2['scan_type']}"
                                )
                                paired_info['create_pix2pix'] = True
                            else:
                                # For separate images (e.g., CycleGAN)
                                paired_info['paired_output_dir'] = os.path.join(
                                    args.output_dir,
                                    subject_id,
                                    vol2['scan_type']
                                )

                            paired_volumes_info.append(paired_info)

        # Use the paired volume info
        if paired_volumes_info:
            volumes_info = paired_volumes_info

    # Process volumes
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            results = list(tqdm(
                executor.map(process_volume, volumes_info),
                total=len(volumes_info),
                desc="Extracting slices"
            ))
    else:
        results = []
        for volume_info in tqdm(volumes_info, desc="Extracting slices"):
            results.append(process_volume(volume_info))

    # Count successes and slices
    successful = [r for r in results if r is not None]
    total_slices = sum(len(r['valid_slices']) for r in successful)

    logger.info(f"Slice extraction completed. {len(successful)}/{len(volumes_info)} volumes processed successfully.")
    logger.info(f"Total slices extracted: {total_slices}")

    # Save results
    results_path = os.path.join(args.output_dir, 'slice_extraction_results.json')
    with open(results_path, 'w') as f:
        json.dump([r for r in results if r is not None], f, indent=2)

    logger.info(f"Results saved to {results_path}")