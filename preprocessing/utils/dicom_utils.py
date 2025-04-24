"""Utilities for DICOM handling in preprocessing pipelines."""

import os
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path


def read_dicom(file_path: str) -> Dict[str, Any]:
    """
    Read a DICOM file and return image data and metadata.

    Args:
        file_path: Path to DICOM file

    Returns:
        Dictionary with image data and metadata
    """
    try:
        import pydicom

        # Read DICOM file
        dcm = pydicom.dcmread(file_path)

        # Extract pixel data
        pixel_array = dcm.pixel_array

        # Apply rescaling if available
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            pixel_array = pixel_array * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

        return {
            'data': pixel_array,
            'metadata': dcm,
            'path': file_path
        }
    except ImportError:
        # Try with SimpleITK if pydicom is not available
        try:
            import SimpleITK as sitk

            reader = sitk.ImageFileReader()
            reader.SetFileName(file_path)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()

            # Load image
            img = reader.Execute()

            # Get metadata dictionary
            metadata = {}
            for k in reader.GetMetaDataKeys():
                metadata[k] = reader.GetMetaData(k)

            # Convert to numpy array
            pixel_array = sitk.GetArrayFromImage(img)[0]  # First slice if 3D

            return {
                'data': pixel_array,
                'sitk_img': img,
                'metadata': metadata,
                'path': file_path
            }
        except ImportError:
            raise ImportError("Neither pydicom nor SimpleITK is available")
        except Exception as e:
            raise ValueError(f"Error reading DICOM file {file_path}: {e}")


def read_dicom_series(directory: str, series_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Read a DICOM series from a directory.

    Args:
        directory: Directory containing DICOM files
        series_id: Optional series ID to load specific series

    Returns:
        Dictionary with volume data and metadata
    """
    try:
        import SimpleITK as sitk

        # Get series IDs in directory
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(directory)

        if not series_IDs:
            raise ValueError(f"No DICOM series found in {directory}")

        # Use specified series ID or first available
        if series_id is not None and series_id in series_IDs:
            selected_series = series_id
        else:
            selected_series = series_IDs[0]

        # Get file names for the series
        dicom_names = reader.GetGDCMSeriesFileNames(directory, selected_series)

        # Read the series
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Convert to numpy
        volume_data = sitk.GetArrayFromImage(image)

        # Get metadata from first slice
        metadata = {}
        if dicom_names:
            single_reader = sitk.ImageFileReader()
            single_reader.SetFileName(dicom_names[0])
            single_reader.LoadPrivateTagsOn()
            single_reader.ReadImageInformation()

            for k in single_reader.GetMetaDataKeys():
                metadata[k] = single_reader.GetMetaData(k)

        return {
            'data': volume_data,
            'sitk_img': image,
            'metadata': metadata,
            'series_id': selected_series,
            'files': dicom_names
        }

    except ImportError:
        # Try with pydicom if SimpleITK is not available
        try:
            import pydicom

            # Find all DICOM files
            dicom_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.dcm') or file.endswith('.DCM'):
                        dicom_files.append(os.path.join(root, file))

            if not dicom_files:
                raise ValueError(f"No DICOM files found in {directory}")

            # Read first file to get series info
            first_dcm = pydicom.dcmread(dicom_files[0])

            # Filter by series ID if provided
            if series_id is not None:
                if hasattr(first_dcm, 'SeriesInstanceUID'):
                    target_series = series_id
                    dicom_files = [f for f in dicom_files if
                                   pydicom.dcmread(f, stop_before_pixels=True).SeriesInstanceUID == target_series]

            # Sort files by position
            slices = [pydicom.dcmread(f) for f in dicom_files]
            slices = sorted(slices, key=lambda x: float(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)

            # Extract pixel data
            volume_data = np.stack([s.pixel_array for s in slices])

            # Apply rescaling if available
            if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
                volume_data = volume_data * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)

            return {
                'data': volume_data,
                'metadata': slices[0],
                'all_slices': slices,
                'files': dicom_files
            }

        except ImportError:
            raise ImportError("Neither SimpleITK nor pydicom is available")
        except Exception as e:
            raise ValueError(f"Error reading DICOM series in {directory}: {e}")


def extract_dicom_metadata(dcm_file: Union[str, Dict[str, Any]],
                           tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract metadata from a DICOM file or object.

    Args:
        dcm_file: Path to DICOM file or dictionary from read_dicom
        tags: Optional list of specific tags to extract

    Returns:
        Dictionary with metadata values
    """
    try:
        import pydicom

        # Get DICOM object
        if isinstance(dcm_file, str):
            dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
        elif isinstance(dcm_file, dict) and 'metadata' in dcm_file:
            if isinstance(dcm_file['metadata'], pydicom.dataset.FileDataset):
                dcm = dcm_file['metadata']
            else:
                # If metadata is a dictionary (from SimpleITK), return it directly
                return dcm_file['metadata']
        else:
            raise ValueError("Invalid input: must be a file path or dictionary from read_dicom")

        # Extract all metadata or specific tags
        metadata = {}

        if tags:
            # Extract specific tags
            for tag in tags:
                if tag in dcm:
                    metadata[tag] = str(dcm.get(tag))
        else:
            # Extract common metadata
            common_tags = [
                'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
                'StudyInstanceUID', 'StudyDate', 'StudyTime', 'StudyDescription',
                'SeriesInstanceUID', 'SeriesNumber', 'SeriesDescription',
                'Modality', 'Manufacturer', 'ManufacturerModelName',
                'SliceThickness', 'SpacingBetweenSlices', 'PixelSpacing',
                'Rows', 'Columns'
            ]

            for tag in common_tags:
                if hasattr(dcm, tag):
                    metadata[tag] = str(getattr(dcm, tag))

        return metadata

    except ImportError:
        # If pydicom is not available, try to parse from metadata dictionary
        if isinstance(dcm_file, dict) and 'metadata' in dcm_file:
            return dcm_file['metadata']
        else:
            raise ImportError("pydicom is required to read DICOM metadata from file")


def get_slice_location(dcm_file: Union[str, Dict[str, Any]]) -> float:
    """
    Get the slice location from a DICOM file or object.

    Args:
        dcm_file: Path to DICOM file or dictionary from read_dicom

    Returns:
        Slice location value
    """
    try:
        import pydicom

        # Get DICOM object
        if isinstance(dcm_file, str):
            dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
        elif isinstance(dcm_file, dict) and 'metadata' in dcm_file:
            if isinstance(dcm_file['metadata'], pydicom.dataset.FileDataset):
                dcm = dcm_file['metadata']
            else:
                # If metadata is a dictionary (from SimpleITK)
                metadata = dcm_file['metadata']
                if '0020|1041' in metadata:  # SliceLocation tag
                    return float(metadata['0020|1041'])
                else:
                    return 0.0
        else:
            raise ValueError("Invalid input: must be a file path or dictionary from read_dicom")

        # Try different methods to get slice location
        if hasattr(dcm, 'SliceLocation'):
            return float(dcm.SliceLocation)
        elif hasattr(dcm, 'ImagePositionPatient'):
            # Use the third value (z-coordinate) of ImagePositionPatient
            return float(dcm.ImagePositionPatient[2])
        elif hasattr(dcm, 'InstanceNumber'):
            # Fall back to InstanceNumber (slice index)
            return float(dcm.InstanceNumber)
        else:
            return 0.0

    except (ImportError, Exception):
        # If pydicom is not available or error occurs
        if isinstance(dcm_file, dict) and 'metadata' in dcm_file:
            metadata = dcm_file['metadata']
            # Try to find slice location in metadata dictionary
            if isinstance(metadata, dict):
                if '0020|1041' in metadata:  # SliceLocation tag
                    return float(metadata['0020|1041'])

        return 0.0


def get_dicom_orientation(dcm_file: Union[str, Dict[str, Any]]) -> Tuple[str, str, str]:
    """
    Get the patient orientation from a DICOM file or object.

    Args:
        dcm_file: Path to DICOM file or dictionary from read_dicom

    Returns:
        Tuple of (row_direction, column_direction, slice_direction)
    """
    try:
        import pydicom

        # Get DICOM object
        if isinstance(dcm_file, str):
            dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
        elif isinstance(dcm_file, dict) and 'metadata' in dcm_file:
            if isinstance(dcm_file['metadata'], pydicom.dataset.FileDataset):
                dcm = dcm_file['metadata']
            else:
                # If metadata is a dictionary (from SimpleITK)
                return ('Unknown', 'Unknown', 'Unknown')
        else:
            raise ValueError("Invalid input: must be a file path or dictionary from read_dicom")

        # Get orientation from ImageOrientationPatient
        if hasattr(dcm, 'ImageOrientationPatient'):
            # ImageOrientationPatient gives the direction cosines for rows and columns
            row_x, row_y, row_z, col_x, col_y, col_z = dcm.ImageOrientationPatient

            # Cross product gives the slice direction
            slice_x = row_y * col_z - row_z * col_y
            slice_y = row_z * col_x - row_x * col_z
            slice_z = row_x * col_y - row_y * col_x

            # Determine orientation labels
            row_dir = get_orientation_label((row_x, row_y, row_z))
            col_dir = get_orientation_label((col_x, col_y, col_z))
            slice_dir = get_orientation_label((slice_x, slice_y, slice_z))

            return (row_dir, col_dir, slice_dir)
        else:
            return ('Unknown', 'Unknown', 'Unknown')

    except (ImportError, Exception):
        return ('Unknown', 'Unknown', 'Unknown')


def get_orientation_label(vector: Tuple[float, float, float]) -> str:
    """
    Convert a direction vector to an orientation label (Left, Right, Anterior, etc.).

    Args:
        vector: Direction vector (x, y, z)

    Returns:
        Orientation label
    """
    # Find the component with the largest absolute value
    abs_vector = [abs(x) for x in vector]
    max_idx = abs_vector.index(max(abs_vector))
    max_val = vector[max_idx]

    # Assign label based on the dominant axis and sign
    if max_idx == 0:  # X-axis
        return 'Right' if max_val < 0 else 'Left'
    elif max_idx == 1:  # Y-axis
        return 'Anterior' if max_val < 0 else 'Posterior'
    else:  # Z-axis
        return 'Inferior' if max_val < 0 else 'Superior'


def sort_dicom_series(dicom_files: List[str]) -> List[str]:
    """
    Sort DICOM files in a series by slice position.

    Args:
        dicom_files: List of DICOM file paths

    Returns:
        Sorted list of DICOM file paths
    """
    # Read slice location/position for each file
    file_positions = []

    for file_path in dicom_files:
        try:
            # Get slice location
            location = get_slice_location(file_path)
            file_positions.append((file_path, location))
        except Exception:
            # Skip files that can't be read or don't have position info
            continue

    # Sort by position
    file_positions.sort(key=lambda x: x[1])

    # Return sorted file paths
    return [fp[0] for fp in file_positions]