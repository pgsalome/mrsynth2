import os
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import cv2

# Import shared utilities
from preprocessing.utils.file_utils import find_nifti_files, ensure_dir


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Extract 2D slices from 3D MRI volumes")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing 3D volumes")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for 2D slices")
    parser.add_argument("--axis", type=int, default=2,
                        choices=[0, 1, 2],
                        help="Axis to extract slices along (0=sagittal, 1=coronal, 2=axial)")
    parser.add_argument("--mask_dir", type=str,
                        help="Directory containing mask volumes for filtering slices")
    parser.add_argument("--content_threshold", type=float, default=0.05,
                        help="Threshold for slice content (fraction of non-zero voxels)")
    parser.add_argument("--slice_range", type=str,
                        help="Range of slices to extract as start,end")
    parser.add_argument("--slice_step", type=int, default=1,
                        help="Step size when extracting slices")
    parser.add_argument("--target_size", type=str, default="256,256",
                        help="Target size for output slices as width,height")
    parser.add_argument("--normalization", type=str, default="min_max",
                        choices=["min_max", "z_score", "percentile", "none"],
                        help="Normalization method for slices")
    parser.add_argument("--output_format", type=str, default="png",
                        choices=["png", "jpg", "npy"],
                        help="Output format for slices")

    return parser.parse_args()


def load_nifti_volume(file_path: str) -> Dict[str, Any]:
    """
    Load a NIfTI volume.

    Args:
        file_path: Path to NIfTI file

    Returns:
        Dictionary with volume data and metadata
    """
    try:
        import nibabel as nib
        img = nib.load(file_path)
        return {
            'data': img.get_fdata(),
            'affine': img.affine,
            'header': img.header,
            'file_format': 'nibabel'
        }
    except ImportError:
        try:
            import SimpleITK as sitk
            img = sitk.ReadImage(file_path)
            data = sitk.GetArrayFromImage(img)
            # SimpleITK uses a different axis ordering (z, y, x) vs (x, y, z)
            data = np.transpose(data, list(range(data.ndim))[::-1])
            return {
                'data': data,
                'sitk_img': img,
                'file_format': 'sitk'
            }
        except ImportError:
            raise ImportError("Neither nibabel nor SimpleITK is available")


def extract_slice(volume_data: np.ndarray, slice_idx: int, axis: int = 0) -> np.ndarray:
    """
    Extract a 2D slice from a 3D volume.

    Args:
        volume_data: 3D volume data
        slice_idx: Slice index
        axis: Axis to extract along

    Returns:
        Extracted 2D slice
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
                           threshold: float = 0.05) -> bool:
    """
    Check if a slice has sufficient content.

    Args:
        slice_data: Slice data
        mask_data: Optional mask data
        threshold: Content threshold

    Returns:
        True if slice has sufficient content
    """
    if mask_data is not None:
        # Use mask to determine content
        content_ratio = np.sum(mask_data > 0) / mask_data.size
    else:
        # Use intensity to determine content
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


def normalize_slice(slice_data: np.ndarray, method: str = 'min_max') -> np.ndarray:
    """
    Normalize a slice.

    Args:
        slice_data: Slice data
        method: Normalization method

    Returns:
        Normalized slice
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

        # Clip data to percentile range
        clipped = np.clip(slice_data, p1, p99)

        # Normalize to [0, 1]
        return (clipped - p1) / (p99 - p1)

    elif method == 'none':
        return slice_data

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_slice(slice_data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a slice.

    Args:
        slice_data: Slice data
        target_size: Target size as (width, height)

    Returns:
        Resized slice
    """
    return cv2.resize(slice_data, target_size, interpolation=cv2.INTER_CUBIC)


def save_slice(slice_data: np.ndarray, output_path: str, output_format: str = 'png') -> None:
    """
    Save a slice.

    Args:
        slice_data: Slice data
        output_path: Output path
        output_format: Output format
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_format == 'npy':
        # Save as NumPy array
        np.save(output_path, slice_data)
    else:
        # Scale to [0, 255] for image formats
        if slice_data.min() < 0 or slice_data.max() > 1:
            slice_data = normalize_slice(slice_data, 'min_max')

        slice_data = (slice_data * 255).astype(np.uint8)

        # Save as image
        cv2.imwrite(output_path, slice_data)


def process_volume(volume_path: str, output_dir: str, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Process a volume and extract slices.

    Args:
        volume_path: Path to volume
        output_dir: Output directory
        args: Command line arguments

    Returns:
        Dictionary with processing information
    """
    # Load volume
    volume_dict = load_nifti_volume(volume_path)
    volume_data = volume_dict['data']

    # Load mask if provided
    mask_data = None
    if args.mask_dir:
        # Construct mask path based on volume path
        volume_filename = os.path.basename(volume_path)
        mask_path = os.path.join(args.mask_dir, volume_filename)

        # Try alternate mask naming if file doesn't exist
        if not os.path.exists(mask_path):
            volume_stem = Path(volume_filename).stem
            mask_path = os.path.join(args.mask_dir, f"{volume_stem}_mask.nii.gz")

        if os.path.exists(mask_path):
            mask_dict = load_nifti_volume(mask_path)
            mask_data = mask_dict['data']

            # Check if mask has the same shape as volume
            if mask_data.shape != volume_data.shape:
                print(f"Warning: Mask shape {mask_data.shape} does not match volume shape {volume_data.shape}")
                mask_data = None

    # Parse target size
    if args.target_size:
        target_size = tuple(map(int, args.target_size.split(',')))
    else:
        # Default to original size
        if args.axis == 0:
            target_size = (volume_data.shape[2], volume_data.shape[1])
        elif args.axis == 1:
            target_size = (volume_data.shape[2], volume_data.shape[0])
        else:  # axis == 2
            target_size = (volume_data.shape[1], volume_data.shape[0])

    # Parse slice range
    if args.slice_range:
        slice_start, slice_end = map(int, args.slice_range.split(','))
    else:
        # Use full range
        if args.axis == 0:
            slice_start, slice_end = 0, volume_data.shape[0]
        elif args.axis == 1:
            slice_start, slice_end = 0, volume_data.shape[1]
        else:  # axis == 2
            slice_start, slice_end = 0, volume_data.shape[2]

    # Create volume output directory
    volume_name = Path(volume_path).stem
    volume_output_dir = os.path.join(output_dir, volume_name)
    ensure_dir(volume_output_dir)

    # Extract slices
    extracted_slices = []

    for slice_idx in range(slice_start, slice_end, args.slice_step):
        # Extract slice
        slice_data = extract_slice(volume_data, slice_idx, args.axis)

        # Extract mask slice if available
        mask_slice = None
        if mask_data is not None:
            mask_slice = extract_slice(mask_data, slice_idx, args.axis)

        # Check content
        if not has_sufficient_content(slice_data, mask_slice, args.content_threshold):
            continue

        # Resize if needed
        if slice_data.shape != target_size[::-1]:  # target_size is (width, height)
            slice_data = resize_slice(slice_data, target_size[::-1])  # resize takes (height, width)

        # Normalize
        if args.normalization != 'none':
            slice_data = normalize_slice(slice_data, args.normalization)

        # Save slice
        output_filename = f"{volume_name}_axis{args.axis}_slice{slice_idx:04d}.{args.output_format}"
        output_path = os.path.join(volume_output_dir, output_filename)
        save_slice(slice_data, output_path, args.output_format)

        extracted_slices.append({
            'slice_idx': slice_idx,
            'output_path': output_path
        })

    return {
        'volume_path': volume_path,
        'output_dir': volume_output_dir,
        'axis': args.axis,
        'slices': extracted_slices
    }


def main():
    args = parse_args()

    # Create output directory
    ensure_dir(args.output_dir)

    # Find NIfTI files
    volume_files = find_nifti_files(args.input_dir)
    print(f"Found {len(volume_files)} NIfTI files")

    # Process volumes
    volume_results = []

    for volume_path in tqdm(volume_files, desc="Processing volumes"):
        try:
            result = process_volume(volume_path, args.output_dir, args)
            volume_results.append(result)
        except Exception as e:
            print(f"Error processing {volume_path}: {e}")

    # Save results
    results = {
        'volumes': volume_results,
        'args': vars(args),
        'num_volumes': len(volume_results),
        'num_slices': sum(len(vol['slices']) for vol in volume_results)
    }

    results_path = os.path.join(args.output_dir, "extraction_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Extraction complete. Extracted {results['num_slices']} slices from {results['num_volumes']} volumes.")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()