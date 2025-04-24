"""Utilities for image registration in preprocessing pipelines."""

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path


def register_volumes(
        fixed_img: Union[str, np.ndarray, Dict[str, Any]],
        moving_img: Union[str, np.ndarray, Dict[str, Any]],
        transform_type: str = 'rigid',
        method: str = 'sitk',
        fixed_mask: Optional[Union[str, np.ndarray]] = None,
        moving_mask: Optional[Union[str, np.ndarray]] = None,
        params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Register one volume to another.

    Args:
        fixed_img: Fixed image (path, array, or dictionary)
        moving_img: Moving image to register (path, array, or dictionary)
        transform_type: Type of transform ('rigid', 'affine', or 'deformable')
        method: Registration method ('sitk' or 'ants')
        fixed_mask: Optional mask for fixed image
        moving_mask: Optional mask for moving image
        params: Additional parameters for registration

    Returns:
        Dictionary with registration results
    """
    if method == 'sitk':
        return _register_sitk(fixed_img, moving_img, transform_type,
                              fixed_mask, moving_mask, params)
    elif method == 'ants':
        return _register_ants(fixed_img, moving_img, transform_type,
                              fixed_mask, moving_mask, params)
    else:
        raise ValueError(f"Unknown registration method: {method}")


def _register_sitk(
        fixed_img: Union[str, np.ndarray, Dict[str, Any]],
        moving_img: Union[str, np.ndarray, Dict[str, Any]],
        transform_type: str = 'rigid',
        fixed_mask: Optional[Union[str, np.ndarray]] = None,
        moving_mask: Optional[Union[str, np.ndarray]] = None,
        params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Register volumes using SimpleITK.

    Args:
        fixed_img: Fixed image (path, array, or dictionary)
        moving_img: Moving image to register (path, array, or dictionary)
        transform_type: Type of transform ('rigid', 'affine', or 'deformable')
        fixed_mask: Optional mask for fixed image
        moving_mask: Optional mask for moving image
        params: Additional parameters for registration

    Returns:
        Dictionary with registration results
    """
    try:
        import SimpleITK as sitk

        # Set default parameters
        if params is None:
            params = {}

        # Load fixed image
        if isinstance(fixed_img, str):
            fixed = sitk.ReadImage(fixed_img)
        elif isinstance(fixed_img, np.ndarray):
            fixed = sitk.GetImageFromArray(np.transpose(fixed_img, axes=range(fixed_img.ndim)[::-1]))
        elif isinstance(fixed_img, dict) and 'sitk_img' in fixed_img:
            fixed = fixed_img['sitk_img']
        else:
            raise ValueError("Unsupported fixed image type")

        # Load moving image
        if isinstance(moving_img, str):
            moving = sitk.ReadImage(moving_img)
        elif isinstance(moving_img, np.ndarray):
            moving = sitk.GetImageFromArray(np.transpose(moving_img, axes=range(moving_img.ndim)[::-1]))
        elif isinstance(moving_img, dict) and 'sitk_img' in moving_img:
            moving = moving_img['sitk_img']
        else:
            raise ValueError("Unsupported moving image type")

        # Load masks if provided
        fixed_mask_img = None
        if fixed_mask is not None:
            if isinstance(fixed_mask, str):
                fixed_mask_img = sitk.ReadImage(fixed_mask)
            elif isinstance(fixed_mask, np.ndarray):
                fixed_mask_img = sitk.GetImageFromArray(np.transpose(fixed_mask, axes=range(fixed_mask.ndim)[::-1]))

        moving_mask_img = None
        if moving_mask is not None:
            if isinstance(moving_mask, str):
                moving_mask_img = sitk.ReadImage(moving_mask)
            elif isinstance(moving_mask, np.ndarray):
                moving_mask_img = sitk.GetImageFromArray(np.transpose(moving_mask, axes=range(moving_mask.ndim)[::-1]))

        # Initialize registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Set metric
        metric = params.get('metric', 'mutual_information')
        if metric == 'mutual_information':
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        elif metric == 'mean_squares':
            registration_method.SetMetricAsMeanSquares()
        elif metric == 'correlation':
            registration_method.SetMetricAsCorrelation()

        # Apply masks if provided
        if fixed_mask_img is not None:
            registration_method.SetMetricFixedMask(fixed_mask_img)
        if moving_mask_img is not None:
            registration_method.SetMetricMovingMask(moving_mask_img)

        # Set optimizer
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=params.get('learning_rate', 1.0),
            numberOfIterations=params.get('num_iterations', 100),
            convergenceMinimumValue=params.get('convergence_min_value', 1e-6),
            convergenceWindowSize=params.get('convergence_window_size', 10)
        )

        # Set interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Create transform based on type
        if transform_type == 'rigid':
            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif transform_type == 'affine':
            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving, sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif transform_type == 'deformable':
            # Start with affine registration
            affine_registration = sitk.ImageRegistrationMethod()
            affine_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            affine_registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
            affine_registration.SetInterpolator(sitk.sitkLinear)

            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving, sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )

            affine_registration.SetInitialTransform(initial_transform, inPlace=False)
            affine_transform = affine_registration.Execute(fixed, moving)

            # Set up for deformable registration
            mesh_size = params.get('mesh_size', [10, 10, 10])
            initial_transform = sitk.BSplineTransformInitializer(
                fixed, mesh_size, order=3
            )

            # Use affine transform first
            registration_method.SetInitialTransform(affine_transform, inPlace=False)

            # Use different optimizer for BSpline
            registration_method.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-5,
                numberOfIterations=params.get('num_iterations', 100),
                maximumNumberOfCorrections=5,
                maximumNumberOfFunctionEvaluations=1000,
                costFunctionConvergenceFactor=1e7
            )
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

        # Set initial transform for rigid and affine
        if transform_type != 'deformable':
            registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Execute registration
        if transform_type != 'deformable':
            transform = registration_method.Execute(fixed, moving)
        else:
            # For deformable, use BSpline
            registration_method.SetInitialTransform(initial_transform, inPlace=True)
            transform = registration_method.Execute(fixed, moving)

        # Apply transform to moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)

        transformed_moving = resampler.Execute(moving)

        # Convert to numpy array
        transformed_moving_array = sitk.GetArrayFromImage(transformed_moving)
        transformed_moving_array = np.transpose(transformed_moving_array,
                                                list(range(transformed_moving_array.ndim))[::-1])

        # Return results
        return {
            'transformed_image': transformed_moving,
            'transformed_array': transformed_moving_array,
            'transform': transform,
            'metric_value': registration_method.GetMetricValue(),
            'iterations': registration_method.GetOptimizerIteration(),
            'stop_condition': registration_method.GetOptimizerStopConditionDescription()
        }

    except ImportError:
        raise ImportError("SimpleITK is required for registration")


def _register_ants(
        fixed_img: Union[str, np.ndarray, Dict[str, Any]],
        moving_img: Union[str, np.ndarray, Dict[str, Any]],
        transform_type: str = 'rigid',
        fixed_mask: Optional[Union[str, np.ndarray]] = None,
        moving_mask: Optional[Union[str, np.ndarray]] = None,
        params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Register volumes using ANTsPy.

    Args:
        fixed_img: Fixed image (path, array, or dictionary)
        moving_img: Moving image to register (path, array, or dictionary)
        transform_type: Type of transform ('rigid', 'affine', or 'deformable')
        fixed_mask: Optional mask for fixed image
        moving_mask: Optional mask for moving image
        params: Additional parameters for registration

    Returns:
        Dictionary with registration results
    """
    try:
        import ants

        # Set default parameters
        if params is None:
            params = {}

        # Load fixed image
        if isinstance(fixed_img, str):
            fixed = ants.image_read(fixed_img)
        elif isinstance(fixed_img, np.ndarray):
            fixed = ants.from_numpy(fixed_img)
        elif isinstance(fixed_img, dict) and 'data' in fixed_img:
            fixed = ants.from_numpy(fixed_img['data'])
        else:
            raise ValueError("Unsupported fixed image type")

        # Load moving image
        if isinstance(moving_img, str):
            moving = ants.image_read(moving_img)
        elif isinstance(moving_img, np.ndarray):
            moving = ants.from_numpy(moving_img)
        elif isinstance(moving_img, dict) and 'data' in moving_img:
            moving = ants.from_numpy(moving_img['data'])
        else:
            raise ValueError("Unsupported moving image type")

        # Load fixed mask if provided
        fixed_mask_img = None
        if fixed_mask is not None:
            if isinstance(fixed_mask, str):
                fixed_mask_img = ants.image_read(fixed_mask)
            elif isinstance(fixed_mask, np.ndarray):
                fixed_mask_img = ants.from_numpy(fixed_mask)

        # Determine transform type for ANTs
        if transform_type == 'rigid':
            ants_transform = 'Rigid'
        elif transform_type == 'affine':
            ants_transform = 'Affine'
        elif transform_type == 'deformable':
            ants_transform = 'SyN'
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

        # Registration parameters
        reg_params = {
            'typeofTransform': ants_transform,
            'initialTransform': None,
            'interpolator': 'linear',
            'verbose': params.get('verbose', False)
        }

        # Set iterations
        if 'num_iterations' in params:
            if isinstance(params['num_iterations'], list):
                reg_params['regIterations'] = params['num_iterations']
            else:
                if ants_transform == 'SyN':
                    reg_params['regIterations'] = [params['num_iterations'] // 4,
                                                   params['num_iterations'] // 2,
                                                   params['num_iterations']]
                else:
                    reg_params['regIterations'] = [params['num_iterations']]

        # Set metric
        if 'metric' in params:
            if params['metric'] == 'mutual_information':
                reg_params['metric'] = 'mattes'
            elif params['metric'] == 'mean_squares':
                reg_params['metric'] = 'meansquares'
            elif params['metric'] == 'correlation':
                reg_params['metric'] = 'gcc'

        # Add mask if provided
        if fixed_mask_img is not None:
            reg_params['mask'] = fixed_mask_img

        # Perform registration
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            **reg_params
        )

        # Extract results
        transformed_moving = registration['warpedmovout']
        transform = registration['fwdtransforms']

        # Convert to numpy array
        transformed_moving_array = transformed_moving.numpy()

        # Return results
        return {
            'transformed_image': transformed_moving,
            'transformed_array': transformed_moving_array,
            'transform': transform,
            'ANTs_dict': registration
        }

    except ImportError:
        raise ImportError("ANTsPy is required for ANTs registration")


def apply_transform(
        img: Union[str, np.ndarray, Dict[str, Any]],
        transform: Any,
        reference_img: Union[str, np.ndarray, Dict[str, Any]],
        method: str = 'sitk',
        interpolation: str = 'linear'
) -> Dict[str, Any]:
    """
    Apply a transform to an image.

    Args:
        img: Image to transform (path, array, or dictionary)
        transform: Transform to apply
        reference_img: Reference image for resampling
        method: Method to use ('sitk' or 'ants')
        interpolation: Interpolation method ('linear', 'nearest', etc.)

    Returns:
        Dictionary with transformed image
    """
    if method == 'sitk':
        return _apply_transform_sitk(img, transform, reference_img, interpolation)
    elif method == 'ants':
        return _apply_transform_ants(img, transform, reference_img, interpolation)
    else:
        raise ValueError(f"Unknown transform method: {method}")


def _apply_transform_sitk(
        img: Union[str, np.ndarray, Dict[str, Any]],
        transform: Any,
        reference_img: Union[str, np.ndarray, Dict[str, Any]],
        interpolation: str = 'linear'
) -> Dict[str, Any]:
    """
    Apply a SimpleITK transform to an image.

    Args:
        img: Image to transform
        transform: SimpleITK transform
        reference_img: Reference image for resampling
        interpolation: Interpolation method

    Returns:
        Dictionary with transformed image
    """
    try:
        import SimpleITK as sitk

        # Load image
        if isinstance(img, str):
            image = sitk.ReadImage(img)
        elif isinstance(img, np.ndarray):
            image = sitk.GetImageFromArray(np.transpose(img, axes=range(img.ndim)[::-1]))
        elif isinstance(img, dict) and 'sitk_img' in img:
            image = img['sitk_img']
        else:
            raise ValueError("Unsupported image type")

        # Load reference image
        if isinstance(reference_img, str):
            reference = sitk.ReadImage(reference_img)
        elif isinstance(reference_img, np.ndarray):
            reference = sitk.GetImageFromArray(np.transpose(reference_img, axes=range(reference_img.ndim)[::-1]))
        elif isinstance(reference_img, dict) and 'sitk_img' in reference_img:
            reference = reference_img['sitk_img']
        else:
            raise ValueError("Unsupported reference image type")

        # Set interpolator
        if interpolation == 'linear':
            interp = sitk.sitkLinear
        elif interpolation == 'nearest':
            interp = sitk.sitkNearestNeighbor
        elif interpolation == 'bspline':
            interp = sitk.sitkBSpline
        else:
            interp = sitk.sitkLinear

        # Apply transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(interp)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)

        transformed = resampler.Execute(image)

        # Convert to numpy array
        transformed_array = sitk.GetArrayFromImage(transformed)
        transformed_array = np.transpose(transformed_array,
                                         list(range(transformed_array.ndim))[::-1])

        return {
            'transformed_image': transformed,
            'transformed_array': transformed_array
        }

    except ImportError:
        raise ImportError("SimpleITK is required for applying SimpleITK transforms")


def _apply_transform_ants(
        img: Union[str, np.ndarray, Dict[str, Any]],
        transform: List[str],
        reference_img: Union[str, np.ndarray, Dict[str, Any]],
        interpolation: str = 'linear'
) -> Dict[str, Any]:
    """
    Apply an ANTs transform to an image.

    Args:
        img: Image to transform
        transform: ANTs transform file paths
        reference_img: Reference image for resampling
        interpolation: Interpolation method

    Returns:
        Dictionary with transformed image
    """
    try:
        import ants

        # Load image
        if isinstance(img, str):
            image = ants.image_read(img)
        elif isinstance(img, np.ndarray):
            image = ants.from_numpy(img)
        elif isinstance(img, dict) and 'data' in img:
            image = ants.from_numpy(img['data'])
        else:
            raise ValueError("Unsupported image type")

        # Load reference image
        if isinstance(reference_img, str):
            reference = ants.image_read(reference_img)
        elif isinstance(reference_img, np.ndarray):
            reference = ants.from_numpy(reference_img)
        elif isinstance(reference_img, dict) and 'data' in reference_img:
            reference = ants.from_numpy(reference_img['data'])
        else:
            raise ValueError("Unsupported reference image type")

        # Apply transform
        transformed = ants.apply_transforms(
            fixed=reference,
            moving=image,
            transformlist=transform,
            interpolator=interpolation
        )

        # Convert to numpy array
        transformed_array = transformed.numpy()

        return {
            'transformed_image': transformed,
            'transformed_array': transformed_array
        }

    except ImportError:
        raise ImportError("ANTsPy is required for applying ANTs transforms")


def create_rigid_transform(
        translation: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float]] = None,
        center: Optional[Tuple[float, float, float]] = None,
        method: str = 'sitk'
) -> Any:
    """
    Create a rigid transform.

    Args:
        translation: Translation vector (x, y, z) in mm
        rotation: Rotation angles (x, y, z) in radians
        center: Center of rotation (x, y, z) in mm
        method: Method to use ('sitk' or 'ants')

    Returns:
        Rigid transform
    """
    if method == 'sitk':
        return _create_rigid_transform_sitk(translation, rotation, center)
    elif method == 'ants':
        raise NotImplementedError("Creating ANTs transforms directly is not yet implemented")
    else:
        raise ValueError(f"Unknown transform method: {method}")


def _create_rigid_transform_sitk(
        translation: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float]] = None,
        center: Optional[Tuple[float, float, float]] = None
) -> Any:
    """
    Create a SimpleITK rigid transform.

    Args:
        translation: Translation vector (x, y, z) in mm
        rotation: Rotation angles (x, y, z) in radians
        center: Center of rotation (x, y, z) in mm

    Returns:
        SimpleITK rigid transform
    """
    try:
        import SimpleITK as sitk

        # Default values
        if translation is None:
            translation = (0.0, 0.0, 0.0)

        if rotation is None:
            rotation = (0.0, 0.0, 0.0)

        # Create transform
        transform = sitk.Euler3DTransform()

        # Set center of rotation if provided
        if center is not None:
            transform.SetCenter(center)

        # Set rotation
        transform.SetRotation(*rotation)

        # Set translation
        transform.SetTranslation(translation)

        return transform

    except ImportError:
        raise ImportError("SimpleITK is required for creating SimpleITK transforms")


def create_affine_transform(
        matrix: Optional[np.ndarray] = None,
        translation: Optional[Tuple[float, float, float]] = None,
        center: Optional[Tuple[float, float, float]] = None,
        method: str = 'sitk'
) -> Any:
    """
    Create an affine transform.

    Args:
        matrix: 3x3 transformation matrix
        translation: Translation vector (x, y, z) in mm
        center: Center of transformation (x, y, z) in mm
        method: Method to use ('sitk' or 'ants')

    Returns:
        Affine transform
    """
    if method == 'sitk':
        return _create_affine_transform_sitk(matrix, translation, center)
    elif method == 'ants':
        raise NotImplementedError("Creating ANTs transforms directly is not yet implemented")
    else:
        raise ValueError(f"Unknown transform method: {method}")


def _create_affine_transform_sitk(
        matrix: Optional[np.ndarray] = None,
        translation: Optional[Tuple[float, float, float]] = None,
        center: Optional[Tuple[float, float, float]] = None
) -> Any:
    """
    Create a SimpleITK affine transform.

    Args:
        matrix: 3x3 transformation matrix
        translation: Translation vector (x, y, z) in mm
        center: Center of transformation (x, y, z) in mm

    Returns:
        SimpleITK affine transform
    """
    try:
        import SimpleITK as sitk

        # Default values
        if matrix is None:
            matrix = np.eye(3)

        if translation is None:
            translation = (0.0, 0.0, 0.0)

        # Create transform
        transform = sitk.AffineTransform(3)

        # Set center if provided
        if center is not None:
            transform.SetCenter(center)

        # Set matrix (convert 3x3 to 9-element vector)
        transform.SetMatrix(matrix.flatten().tolist())

        # Set translation
        transform.SetTranslation(translation)

        return transform

    except ImportError:
        raise ImportError("SimpleITK is required for creating SimpleITK transforms")


def evaluate_registration(
        fixed_img: Union[str, np.ndarray, Dict[str, Any]],
        registered_img: Union[str, np.ndarray, Dict[str, Any]],
        metric: str = 'mse'
) -> float:
    """
    Evaluate the quality of registration.

    Args:
        fixed_img: Fixed image (path, array, or dictionary)
        registered_img: Registered image (path, array, or dictionary)
        metric: Metric to use ('mse', 'ncc', 'mi')

    Returns:
        Metric value
    """
    # Load fixed image
    if isinstance(fixed_img, str):
        fixed = np.array(load_image(fixed_img))
    elif isinstance(fixed_img, np.ndarray):
        fixed = fixed_img
    elif isinstance(fixed_img, dict) and 'data' in fixed_img:
        fixed = fixed_img['data']
    else:
        raise ValueError("Unsupported fixed image type")

    # Load registered image
    if isinstance(registered_img, str):
        registered = np.array(load_image(registered_img))
    elif isinstance(registered_img, np.ndarray):
        registered = registered_img
    elif isinstance(registered_img, dict) and 'data' in registered_img:
        registered = registered_img['data']
    elif isinstance(registered_img, dict) and 'transformed_array' in registered_img:
        registered = registered_img['transformed_array']
    else:
        raise ValueError("Unsupported registered image type")

    # Ensure same shape
    if fixed.shape != registered.shape:
        raise ValueError(f"Images have different shapes: {fixed.shape} vs {registered.shape}")

    # Calculate metric
    if metric == 'mse':
        # Mean squared error (lower is better)
        return np.mean((fixed - registered) ** 2)

    elif metric == 'ncc':
        # Normalized cross-correlation (higher is better)
        fixed_norm = (fixed - np.mean(fixed)) / (np.std(fixed) * fixed.size)
        registered_norm = (registered - np.mean(registered)) / np.std(registered)
        return np.sum(fixed_norm * registered_norm)

    elif metric == 'mi':
        # Mutual information (higher is better)
        try:
            from sklearn.metrics import mutual_info_score

            # Flatten and discretize images
            fixed_flat = fixed.flatten()
            registered_flat = registered.flatten()

            # Scale to integers for histogram binning
            fixed_scaled = (fixed_flat * 255).astype(np.uint8)
            registered_scaled = (registered_flat * 255).astype(np.uint8)

            # Calculate mutual information
            return mutual_info_score(fixed_scaled, registered_scaled)

        except ImportError:
            raise ImportError("scikit-learn is required for mutual information calculation")

    else:
        raise ValueError(f"Unknown metric: {metric}")


# Helper function for loading an image (used in evaluate_registration)
def load_image(file_path: str) -> np.ndarray:
    """
    Load an image from file.

    Args:
        file_path: Path to image file

    Returns:
        Loaded image as numpy array
    """
    try:
        # Try loading as a medical image first
        try:
            import SimpleITK as sitk
            img = sitk.ReadImage(file_path)
            return sitk.GetArrayFromImage(img)
        except:
            # If it fails, try with a regular image library
            try:
                from PIL import Image
                return np.array(Image.open(file_path))
            except:
                import cv2
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Could not load image {file_path}")
                return img
    except Exception as e:
        raise ValueError(f"Could not load image {file_path}: {e}")