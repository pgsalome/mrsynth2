import os
import glob
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any


def find_files(directory: str, pattern: str = "*", recursive: bool = True) -> List[str]:
    """
    Find files matching a pattern in a directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively

    Returns:
        List of file paths
    """
    if recursive:
        return glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    else:
        return glob.glob(os.path.join(directory, pattern))


def find_image_files(directory: str, extensions: Optional[List[str]] = None,
                     recursive: bool = True) -> List[str]:
    """
    Find image files in a directory.

    Args:
        directory: Directory to search
        extensions: List of image extensions to find
        recursive: Whether to search recursively

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    image_files = []
    for ext in extensions:
        # Handle case sensitivity
        files = find_files(directory, f"*{ext}", recursive)
        files.extend(find_files(directory, f"*{ext.upper()}", recursive))
        image_files.extend(files)

    return sorted(image_files)


def find_dicom_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Find DICOM files in a directory.

    Args:
        directory: Directory to search
        recursive: Whether to search recursively

    Returns:
        List of DICOM file paths
    """
    dicom_files = []

    # Look for .dcm extension
    dicom_files.extend(find_files(directory, "*.dcm", recursive))
    dicom_files.extend(find_files(directory, "*.DCM", recursive))

    # Look for DICOM files without extension
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if not os.path.splitext(file)[1]:  # No extension
                    file_path = os.path.join(root, file)
                    # Check if it might be a DICOM file
                    try:
                        with open(file_path, 'rb') as f:
                            # Check for DICOM magic bytes
                            f.seek(128)
                            if f.read(4) == b'DICM':
                                dicom_files.append(file_path)
                    except:
                        pass

    return sorted(dicom_files)


def find_nifti_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Find NIfTI files in a directory.

    Args:
        directory: Directory to search
        recursive: Whether to search recursively

    Returns:
        List of NIfTI file paths
    """
    nifti_files = []

    # Look for .nii and .nii.gz extensions
    nifti_files.extend(find_files(directory, "*.nii", recursive))
    nifti_files.extend(find_files(directory, "*.nii.gz", recursive))

    return sorted(nifti_files)


def match_files_by_pattern(files: List[str], pattern: str) -> Dict[str, str]:
    """
    Match files using a regex pattern with a capturing group.

    Args:
        files: List of file paths
        pattern: Regex pattern with a capturing group for matching

    Returns:
        Dictionary mapping matched key to file path
    """
    matches = {}
    regex = re.compile(pattern)

    for file in files:
        match = regex.search(os.path.basename(file))
        if match and match.groups():
            key = match.group(1)
            matches[key] = file

    return matches


def pair_files_by_name(files_a: List[str], files_b: List[str]) -> List[Tuple[str, str]]:
    """
    Pair files from two lists by basename (without extension).

    Args:
        files_a: First list of file paths
        files_b: Second list of file paths

    Returns:
        List of paired file paths
    """
    # Create dictionaries with basenames as keys
    a_dict = {Path(f).stem: f for f in files_a}
    b_dict = {Path(f).stem: f for f in files_b}

    # Find common keys
    common_keys = set(a_dict.keys()).intersection(set(b_dict.keys()))

    # Create pairs
    pairs = [(a_dict[key], b_dict[key]) for key in sorted(common_keys)]

    return pairs


def create_dir_structure(base_dir: str, structure: Dict[str, Any]) -> None:
    """
    Create a directory structure.

    Args:
        base_dir: Base directory
        structure: Directory structure as nested dictionary
    """
    os.makedirs(base_dir, exist_ok=True)

    for key, value in structure.items():
        path = os.path.join(base_dir, key)

        if isinstance(value, dict):
            # Recursively create subdirectories
            create_dir_structure(path, value)
        else:
            # Create directory
            os.makedirs(path, exist_ok=True)


def ensure_dir(path: str) -> str:
    """
    Ensure a directory exists.

    Args:
        path: Directory path

    Returns:
        Created directory path
    """
    os.makedirs(path, exist_ok=True)
    return path