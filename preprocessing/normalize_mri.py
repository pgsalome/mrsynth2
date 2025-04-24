#!/usr/bin/env python
"""
Normalize MRI data

This script applies intensity normalization to MRI data to standardize
intensity values across different scans and subjects. It supports various
normalization methods including z-score, min-max, percentile-based,
histogram matching, and N4 bias field correction.
"""

import os
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

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

# For N4 bias field correction
try:
    from src.preprocessing.normalizers import N4BiasFieldCorrection

    HAS_N4 = True
except ImportError:
    try:
        import SimpleITK as sitk

        HAS_N4 = HAS_SITK
    except ImportError:
        HAS_N4 = False
        logging.warning("N4 bias field correction unavailable. Install SimpleITK for this functionality.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_nifti(file_path):
    """Load a NIfTI file using either nibabel or SimpleITK"""
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


def save_nifti(file_path, img_dict):
    """Save a NIfTI file using either nibabel or SimpleITK"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if img_dict['file_format'] == 'nibabel':
        nib.save(nib.Nifti1Image(img_dict['data'], img_dict['affine'], header=img_dict['header']), file_path)
    elif img_dict['file_format'] == 'sitk':
        # Convert back to SimpleITK's axis ordering
        data = np.transpose(img_dict['data'], list(range(img_dict['data'].ndim))[::-1])
        img = sitk.GetImageFromArray(data)
        img.CopyInformation(img_dict['sitk_img'])
        sitk.WriteImage(img, file_path)
    else:
        raise ValueError(f"Unknown file format: {img_dict['file_format']}")

    return file_path


def apply_mask(data, mask_data):
    """Apply a mask to the data"""
    if mask_data.shape != data.shape:
        raise ValueError(f"Mask shape {mask_data.shape} does not match data shape {data.shape}")

    # Create boolean mask
    bool_mask = mask_data > 0

    # Apply mask
    masked_data = data.copy()
    masked_data[~bool_mask] = 0

    return masked_data


def normalize_z_score(data, mask=None):
    """
    Z-score normalization: (data - mean) / std

    Args:
        data: Input data array
        mask: Optional mask to apply before calculating statistics

    Returns:
        Normalized data
    """
    if mask is not None:
        # Calculate statistics only on masked region
        masked_data = data[mask > 0]
        mean = np.mean(masked_data)
        std = np.std(masked_data)
    else:
        # Calculate statistics on entire volume
        mean = np.mean(data)
        std = np.std(data)

    # Avoid division by zero
    if std == 0:
        logger.warning("Standard deviation is zero, returning zeros")
        return np.zeros_like(data)

    return (data - mean) / std


def normalize_min_max(data, mask=None, new_min=0, new_max=1):
    """
    Min-max normalization: (data - min) / (max - min) * (new_max - new_min) + new_min

    Args:
        data: Input data array
        mask: Optional mask to apply before calculating statistics
        new_min: Minimum value in output range
        new_max: Maximum value in output range

    Returns:
        Normalized data
    """
    if mask is not None:
        # Calculate statistics only on masked region
        masked_data = data[mask > 0]
        data_min = np.min(masked_data)
        data_max = np.max(masked_data)
    else:
        # Calculate statistics on entire volume
        data_min = np.min(data)
        data_max = np.max(data)

    # Avoid division by zero
    if data_max == data_min:
        logger.warning("Max and min are equal, returning constant value")
        return np.ones_like(data) * (new_max + new_min) / 2

    return (data - data_min) / (data_max - data_min) * (new_max - new_min) + new_min


def normalize_percentile(data, mask=None, percentile_low=1, percentile_high=99, new_min=0, new_max=1):
    """
    Percentile-based normalization to handle outliers

    Args:
        data: Input data array
        mask: Optional mask to apply before calculating statistics
        percentile_low: Lower percentile (e.g., 1st percentile)
        percentile_high: Higher percentile (e.g., 99th percentile)
        new_min: Minimum value in output range
        new_max: Maximum value in output range

    Returns:
        Normalized data
    """
    if mask is not None:
        # Calculate percentiles only on masked region
        masked_data = data[mask > 0]
        p_low = np.percentile(masked_data, percentile_low)
        p_high = np.percentile(masked_data, percentile_high)
    else:
        # Calculate percentiles on entire volume
        p_low = np.percentile(data, percentile_low)
        p_high = np.percentile(data, percentile_high)

    # Clip data to the percentile range
    clipped_data = np.clip(data, p_low, p_high)

    # Avoid division by zero
    if p_high == p_low:
        logger.warning("Percentile high and low are equal, returning constant value")
        return np.ones_like(data) * (new_max + new_min) / 2

    # Normalize to the new range
    return (clipped_data - p_low) / (p_high - p_low) * (new_max - new_min) + new_min


def normalize_histogram_matching(data, reference_data, mask=None, ref_mask=None, n_bins=256):
    """
    Match histogram of data to reference_data

    Args:
        data: Input data array
        reference_data: Reference data array to match histogram to
        mask: Optional mask for input data
        ref_mask: Optional mask for reference data
        n_bins: Number of bins for histogram

    Returns:
        Normalized data with histogram matched to reference
    """
    # Apply masks if provided
    if mask is not None:
        data_values = data[mask > 0]
    else:
        data_values = data.flatten()

    if ref_mask is not None:
        ref_values = reference_data[ref_mask > 0]
    else:
        ref_values = reference_data.flatten()

    # Get histograms
    hist, bin_edges = np.histogram(data_values, bins=n_bins, density=True)
    ref_hist, ref_bin_edges = np.histogram(ref_values, bins=n_bins, density=True)

    # Get cumulative distributions
    cdf = hist.cumsum() / hist.sum()
    ref_cdf = ref_hist.cumsum() / ref_hist.sum()

    # Get bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ref_bin_centers = (ref_bin_edges[:-1] + ref_bin_edges[1:]) / 2

    # Create interpolation function
    from scipy import interpolate
    interp_ref_values = interpolate.interp1d(ref_cdf, ref_bin_centers, bounds_error=False,
                                             fill_value=(ref_bin_centers[0], ref_bin_centers[-1]))

    # Map each data value to the corresponding reference value
    interp_vals = interp_ref_values(cdf)

    # Map input values to matched values
    # Find which bin each value is in
    indices = np.searchsorted(bin_centers, data)
    indices = np.clip(indices, 0, len(interp_vals) - 1)

    # Return the reference value for that bin
    matched_data = np.zeros_like(data)
    matched_data = interp_vals[indices]

    return matched_data


def apply_n4_bias_field_correction(file_path, output_path, mask_path=None):
    """
    Apply N4 bias field correction to a NIfTI file

    Args:
        file_path: Path to input NIfTI file
        output_path: Path to output NIfTI file
        mask_path: Optional path to mask NIfTI file

    Returns:
        Path to output NIfTI file
    """
    if not HAS_N4:
        raise ImportError("N4 bias field correction not available")

    if hasattr(N4BiasFieldCorrection, 'correct'):
        # Use our implementation if available
        corrector = N4BiasFieldCorrection()
        return corrector.correct(file_path, output_path, mask_path)
    elif HAS_SITK:
        # Use SimpleITK implementation
        img = sitk.ReadImage(file_path)

        # Apply mask if provided
        if mask_path:
            mask = sitk.ReadImage(mask_path)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrected = corrector.Execute(img, mask)
        else:
            # Create a simple mask - assume non-zero values are foreground
            mask = sitk.BinaryThreshold(img, lowerThreshold=1, upperThreshold=float('inf'), insideValue=1,
                                        outsideValue=0)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrected = corrector.Execute(img, mask)

        sitk.WriteImage(corrected, output_path)
        return output_path
    else:
        raise ImportError("N4 bias field correction not available")


def process_file(file_info):
    """Process a single file with the specified normalizations"""
    input_file = file_info['input_file']
    output_file = file_info['output_file']
    params = file_info['params']

    try:
        # Load input file
        img_dict = load_nifti(input_file)
        data = img_dict['data']

        # Load mask if provided
        mask = None
        if params.get('mask_file'):
            mask_dict = load_nifti(params['mask_file'])
            mask = mask_dict['data']

        # Apply N4 bias field correction if requested
        if params.get('apply_n4', False) and HAS_N4:
            n4_output = os.path.join(os.path.dirname(output_file), f"{Path(output_file).stem}_n4.nii.gz")
            apply_n4_bias_field_correction(input_file, n4_output, params.get('mask_file'))

            # Load the N4-corrected image
            img_dict = load_nifti(n4_output)
            data = img_dict['data']

        # Apply normalization based on method
        method = params.get('method', 'z_score')

        if method == 'z_score':
            data = normalize_z_score(data, mask)
        elif method == 'min_max':
            data = normalize_min_max(data, mask,
                                     params.get('new_min', 0),
                                     params.get('new_max', 1))
        elif method == 'percentile':
            data = normalize_percentile(data, mask,
                                        params.get('percentile_low', 1),
                                        params.get('percentile_high', 99),
                                        params.get('new_min', 0),
                                        params.get('new_max', 1))
        elif method == 'histogram_matching' and params.get('reference_file'):
            # Load reference image
            ref_dict = load_nifti(params['reference_file'])
            ref_data = ref_dict['data']

            # Load reference mask if provided
            ref_mask = None
            if params.get('reference_mask'):
                ref_mask_dict = load_nifti(params['reference_mask'])
                ref_mask = ref_mask_dict['data']

            data = normalize_histogram_matching(data, ref_data, mask, ref_mask,
                                                params.get('n_bins', 256))
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return None

        # Update data in image dictionary
        img_dict['data'] = data

        # Save normalized image
        save_nifti(output_file, img_dict)

        return output_file

    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")
        return None


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Normalize MRI data")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing NIfTI files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for normalized NIfTI files")
    parser.add_argument("--files_list", type=str,
                        help="JSON file with list of files to process (optional)")
    parser.add_argument("--method", type=str, default="z_score",
                        choices=["z_score", "min_max", "percentile", "histogram_matching"],
                        help="Normalization method")
    parser.add_argument("--mask_dir", type=str,
                        help="Directory containing mask files (optional)")
    parser.add_argument("--reference_file", type=str,
                        help="Reference file for histogram matching (optional)")
    parser.add_argument("--reference_mask", type=str,
                        help="Mask file for reference in histogram matching (optional)")
    parser.add_argument("--new_min", type=float, default=0,
                        help="Minimum value after normalization")
    parser.add_argument("--new_max", type=float, default=1,
                        help="Maximum value after normalization")
    parser.add_argument("--percentile_low", type=float, default=1,
                        help="Lower percentile for percentile normalization")
    parser.add_argument("--percentile_high", type=float, default=99,
                        help="Higher percentile for percentile normalization")
    parser.add_argument("--apply_n4", action="store_true",
                        help="Apply N4 bias field correction before normalization")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of files to process
    if args.files_list:
        # Load from JSON file
        with open(args.files_list, 'r') as f:
            files_info = json.load(f)
    else:
        # Auto-find NIfTI files in input directory
        input_files = []
        for root, _, files in os.walk(args.input_dir):
            for f in files:
                if f.endswith(('.nii', '.nii.gz')):
                    input_files.append(os.path.join(root, f))

        # Create file info dictionaries
        files_info = []
        for input_file in input_files:
            # Determine output file path
            rel_path = os.path.relpath(input_file, args.input_dir)
            output_file = os.path.join(args.output_dir, rel_path)

            # Determine mask file path if mask directory provided
            mask_file = None
            if args.mask_dir:
                mask_rel_path = rel_path
                mask_file = os.path.join(args.mask_dir, mask_rel_path)
                if not os.path.exists(mask_file):
                    # Try with _mask suffix
                    mask_name = Path(mask_rel_path).stem + '_mask.nii.gz'
                    mask_file = os.path.join(args.mask_dir, os.path.dirname(mask_rel_path), mask_name)
                    if not os.path.exists(mask_file):
                        mask_file = None

            files_info.append({
                'input_file': input_file,
                'output_file': output_file,
                'params': {
                    'method': args.method,
                    'mask_file': mask_file,
                    'reference_file': args.reference_file,
                    'reference_mask': args.reference_mask,
                    'new_min': args.new_min,
                    'new_max': args.new_max,
                    'percentile_low': args.percentile_low,
                    'percentile_high': args.percentile_high,
                    'apply_n4': args.apply_n4
                }
            })

    # Process files
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            results = list(tqdm(
                executor.map(process_file, files_info),
                total=len(files_info),
                desc="Normalizing MRI data"
            ))
    else:
        results = []
        for file_info in tqdm(files_info, desc="Normalizing MRI data"):
            results.append(process_file(file_info))

    # Count successes
    successful = [r for r in results if r is not None]
    logger.info(f"Normalization completed. {len(successful)}/{len(files_info)} files processed successfully.")