#!/usr/bin/env python
"""
Convert DICOM files to NIfTI format

This script converts DICOM files from a source directory into NIfTI format
and saves them to a destination directory. It can handle multiple subjects
and scan types/sequences.
"""

import os
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
from tqdm import tqdm

# Try different DICOM to NIfTI converters
try:
    import dicom2nifti

    CONVERTER = 'dicom2nifti'
except ImportError:
    try:
        import nibabel as nib
        import pydicom

        CONVERTER = 'nibabel'
    except ImportError:
        try:
            import SimpleITK as sitk

            CONVERTER = 'sitk'
        except ImportError:
            raise ImportError(
                "Could not find any DICOM converter. Please install one of: dicom2nifti, nibabel+pydicom, or SimpleITK")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_with_dicom2nifti(dicom_dir, output_file):
    """Convert DICOM to NIfTI using dicom2nifti library"""
    dicom2nifti.convert_directory(dicom_dir, output_file, compression=True, reorient=True)
    return output_file


def convert_with_nibabel(dicom_dir, output_file):
    """Convert DICOM to NIfTI using nibabel and pydicom"""
    # Get all DICOM files in the directory
    dicom_files = sorted([os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
                          if f.endswith('.dcm') or f.endswith('.DCM')])

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    # Read the first DICOM file to get the initial dataset
    first_dicom = pydicom.dcmread(dicom_files[0])

    # Check if it's a valid DICOM file with pixel data
    if not hasattr(first_dicom, 'pixel_array'):
        raise ValueError(f"DICOM file {dicom_files[0]} does not contain pixel data")

    # Create numpy array for pixel data
    import numpy as np
    if len(dicom_files) > 1:
        # Multi-slice case
        slices = [pydicom.dcmread(f) for f in dicom_files]

        # Sort by slice location if available
        try:
            slices = sorted(slices, key=lambda x: float(x.SliceLocation))
        except (AttributeError, ValueError):
            try:
                slices = sorted(slices, key=lambda x: float(x.InstanceNumber))
            except (AttributeError, ValueError):
                logger.warning("Could not sort slices by SliceLocation or InstanceNumber")

        # Extract pixel data from each slice
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.insert(0, len(slices))
        img3d = np.zeros(img_shape, dtype=slices[0].pixel_array.dtype)

        # Fill 3D array with the images from the slices
        for i, s in enumerate(slices):
            img2d = s.pixel_array
            img3d[i, :, :] = img2d
    else:
        # Single slice case
        img3d = first_dicom.pixel_array
        if img3d.ndim == 2:
            img3d = img3d.reshape(1, *img3d.shape)

    # Get spacing information
    try:
        dx, dy = first_dicom.PixelSpacing
    except (AttributeError, ValueError):
        dx = dy = 1.0
        logger.warning("Could not read PixelSpacing, using default values")

    try:
        dz = first_dicom.SliceThickness
    except (AttributeError, ValueError):
        dz = 1.0
        logger.warning("Could not read SliceThickness, using default value")

    # Create NIfTI object
    affine = np.eye(4)
    affine[0, 0] = dx
    affine[1, 1] = dy
    affine[2, 2] = dz

    nii_img = nib.Nifti1Image(img3d, affine)

    # Save NIfTI file
    nib.save(nii_img, output_file)
    return output_file


def convert_with_sitk(dicom_dir, output_file):
    """Convert DICOM to NIfTI using SimpleITK"""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)

    if not dicom_names:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    sitk.WriteImage(image, output_file)
    return output_file


def convert_directory(source_dir, output_dir, subject_id=None, scan_type=None):
    """
    Convert a directory of DICOM files to NIfTI format

    Args:
        source_dir: Source directory containing DICOM files
        output_dir: Output directory for NIfTI files
        subject_id: Subject ID (optional, for naming)
        scan_type: Scan type or sequence (optional, for naming)

    Returns:
        Path to the output NIfTI file
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if subject_id and scan_type:
        output_file = output_dir / f"{subject_id}_{scan_type}.nii.gz"
    elif subject_id:
        output_file = output_dir / f"{subject_id}.nii.gz"
    else:
        output_file = output_dir / f"{source_dir.name}.nii.gz"

    # Check if file already exists
    if output_file.exists():
        logger.info(f"File {output_file} already exists, skipping conversion")
        return str(output_file)

    # Convert based on available converter
    try:
        if CONVERTER == 'dicom2nifti':
            return convert_with_dicom2nifti(str(source_dir), str(output_file))
        elif CONVERTER == 'nibabel':
            return convert_with_nibabel(str(source_dir), str(output_file))
        elif CONVERTER == 'sitk':
            return convert_with_sitk(str(source_dir), str(output_file))
    except Exception as e:
        logger.error(f"Error converting {source_dir}: {e}")
        return None

    return str(output_file)


def process_subject(subject_info):
    """Process a single subject from the subject_info dictionary"""
    subject_id = subject_info['subject_id']
    scans = subject_info['scans']

    results = {}
    for scan_type, scan_dir in scans.items():
        output_file = convert_directory(
            scan_dir,
            os.path.join(args.output_dir, subject_id),
            subject_id=subject_id,
            scan_type=scan_type
        )
        if output_file:
            results[scan_type] = output_file

    return subject_id, results


def find_dicom_dirs(base_dir, recursive=True, max_depth=3):
    """Find directories containing DICOM files"""
    base_dir = Path(base_dir)
    dicom_dirs = []

    def is_dicom_dir(dir_path, min_files=10):
        """Check if a directory contains DICOM files"""
        dicom_count = sum(1 for f in os.listdir(dir_path)
                          if f.endswith(('.dcm', '.DCM')))
        return dicom_count >= min_files

    def scan_dir(current_dir, current_depth=0):
        """Recursively scan directory for DICOM files"""
        if current_depth > max_depth:
            return

        try:
            # Check if current directory contains DICOM files
            if is_dicom_dir(current_dir):
                dicom_dirs.append(current_dir)
                return  # Don't need to go deeper

            # Check subdirectories if recursive
            if recursive:
                for item in os.listdir(current_dir):
                    item_path = current_dir / item
                    if item_path.is_dir():
                        scan_dir(item_path, current_depth + 1)
        except (PermissionError, FileNotFoundError) as e:
            logger.warning(f"Could not access {current_dir}: {e}")

    scan_dir(base_dir)
    return dicom_dirs


def auto_detect_subjects(base_dir, recursive=True):
    """
    Automatically detect subjects and scan types in a directory structure

    Attempts to identify a directory structure like:
    base_dir/
      subject1/
        T1/
          dicom_files...
        T2/
          dicom_files...
      subject2/
        ...

    Or:
    base_dir/
      subject1_T1/
        dicom_files...
      subject1_T2/
        dicom_files...
      ...

    Returns:
        List of dictionaries with subject_id and scans
    """
    base_dir = Path(base_dir)
    subjects = []

    # First approach: Find directories with DICOM files
    dicom_dirs = find_dicom_dirs(base_dir, recursive)

    if not dicom_dirs:
        logger.warning(f"No DICOM directories found in {base_dir}")
        return []

    # Try to identify if we have subject/scan_type directory structure
    # or subject_scan_type directory naming

    # Check for subject/scan_type structure
    potential_subject_dirs = set(d.parent for d in dicom_dirs)

    if len(potential_subject_dirs) > 0 and all(d.parent == base_dir for d in potential_subject_dirs):
        # We have base_dir/subject/scan_type structure
        for subject_dir in potential_subject_dirs:
            subject_id = subject_dir.name
            scans = {}

            # Find scan directories under this subject
            for dicom_dir in dicom_dirs:
                if dicom_dir.parent == subject_dir:
                    scan_type = dicom_dir.name
                    scans[scan_type] = str(dicom_dir)

            if scans:
                subjects.append({
                    'subject_id': subject_id,
                    'scans': scans
                })
    else:
        # Assume we have base_dir/subject_scan_type structure
        # Try to extract subject and scan info from directory names
        from collections import defaultdict
        subject_scans = defaultdict(dict)

        for dicom_dir in dicom_dirs:
            # Try common naming patterns
            dir_name = dicom_dir.name

            # Pattern: subject_scantype
            parts = dir_name.split('_', 1)
            if len(parts) == 2:
                subject_id, scan_type = parts
                subject_scans[subject_id][scan_type] = str(dicom_dir)
                continue

            # Pattern: sub-id_seq-type
            if 'sub-' in dir_name and 'seq-' in dir_name:
                parts = dir_name.split('_')
                subject_id = next((p.replace('sub-', '') for p in parts if p.startswith('sub-')), None)
                scan_type = next((p.replace('seq-', '') for p in parts if p.startswith('seq-')), None)
                if subject_id and scan_type:
                    subject_scans[subject_id][scan_type] = str(dicom_dir)
                    continue

            # If no pattern matches, use directory name as both subject and scan
            subject_scans[dir_name]['unknown'] = str(dicom_dir)

        # Convert to list of dictionaries
        for subject_id, scans in subject_scans.items():
            subjects.append({
                'subject_id': subject_id,
                'scans': scans
            })

    return subjects


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Convert DICOM files to NIfTI format")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing DICOM files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for NIfTI files")
    parser.add_argument("--subjects_file", type=str,
                        help="JSON file with subject and scan information (optional)")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively search for DICOM files in subdirectories")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get subject information
    if args.subjects_file:
        # Load from JSON file
        with open(args.subjects_file, 'r') as f:
            subjects = json.load(f)
    else:
        # Auto-detect subjects
        logger.info(f"Auto-detecting subjects in {args.input_dir}")
        subjects = auto_detect_subjects(args.input_dir, args.recursive)

        if subjects:
            logger.info(f"Found {len(subjects)} subjects")
            # Save detected subjects for future use
            with open(os.path.join(args.output_dir, 'subjects.json'), 'w') as f:
                json.dump(subjects, f, indent=2)
        else:
            logger.error("No subjects detected. Please check the input directory or provide a subjects file.")
            exit(1)

    # Process subjects
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            results = list(tqdm(
                executor.map(process_subject, subjects),
                total=len(subjects),
                desc="Converting DICOM to NIfTI"
            ))
    else:
        results = []
        for subject_info in tqdm(subjects, desc="Converting DICOM to NIfTI"):
            results.append(process_subject(subject_info))

    # Save results
    results_dict = {subject_id: scan_files for subject_id, scan_files in results if scan_files}
    with open(os.path.join(args.output_dir, 'conversion_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Conversion completed. Results saved to {os.path.join(args.output_dir, 'conversion_results.json')}")