"""
MRI registration methods

This module provides methods for registering MRI images, including rigid, affine,
and deformable registration.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List

try:
    import SimpleITK as sitk

    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import ants

    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False

# Set up logging
import logging

logger = logging.getLogger(__name__)


class RegistrationMethod:
    """Base class for registration methods"""

    def __init__(self, **kwargs):
        """Initialize the registration method"""
        self.kwargs = kwargs

    def register(self, fixed_image: Union[str, np.ndarray], moving_image: Union[str, np.ndarray],
                 fixed_mask: Optional[Union[str, np.ndarray]] = None,
                 moving_mask: Optional[Union[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Register moving image to fixed image

        Args:
            fixed_image: Fixed/reference image (path or array)
            moving_image: Moving image to be registered (path or array)
            fixed_mask: Optional mask for fixed image (path or array)
            moving_mask: Optional mask for moving image (path or array)

        Returns:
            Dictionary with registration results
        """
        raise NotImplementedError("Subclasses must implement this method")


class SITKRegistration(RegistrationMethod):
    """
    SimpleITK-based registration

    This class provides registration methods using SimpleITK.
    """

    def __init__(self, transform_type: str = 'rigid', **kwargs):
        """
        Initialize the SimpleITK registration method

        Args:
            transform_type: Type of transformation ('rigid', 'affine', 'deformable')
            **kwargs: Additional parameters for registration
        """
        if not HAS_SITK:
            raise ImportError("SimpleITK is required for SITKRegistration")

        super().__init__(**kwargs)
        self.transform_type = transform_type

    def register(self, fixed_image: Union[str, np.ndarray], moving_image: Union[str, np.ndarray],
                 fixed_mask: Optional[Union[str, np.ndarray]] = None,
                 moving_mask: Optional[Union[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Register moving image to fixed image

        Args:
            fixed_image: Fixed/reference image (path or array)
            moving_image: Moving image to be registered (path or array)
            fixed_mask: Optional mask for fixed image (path or array)
            moving_mask: Optional mask for moving image (path or array)

        Returns:
            Dictionary with registration results
        """
        # Load images
        fixed = self._load_image(fixed_image)
        moving = self._load_image(moving_image)

        # Load masks if provided
        fixed_mask_img = None
        if fixed_mask is not None:
            fixed_mask_img = self._load_image(fixed_mask)

        moving_mask_img = None
        if moving_mask is not None:
            moving_mask_img = self._load_image(moving_mask)

        # Initialize registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Set up metric
        metric = self.kwargs.get('metric', 'mutual_information')
        if metric == 'mutual_information':
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        elif metric == 'mean_squares':
            registration_method.SetMetricAsMeanSquares()
        elif metric == 'correlation':
            registration_method.SetMetricAsCorrelation()
        else:
            # Default to mutual information
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

        # Apply masks if provided
        if fixed_mask_img is not None:
            registration_method.SetMetricFixedMask(fixed_mask_img)
        if moving_mask_img is not None:
            registration_method.SetMetricMovingMask(moving_mask_img)

        # Set optimizer
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=self.kwargs.get('learning_rate', 1.0),
            numberOfIterations=self.kwargs.get('num_iterations', 100),
            convergenceMinimumValue=self.kwargs.get('convergence_min_value', 1e-6),
            convergenceWindowSize=self.kwargs.get('convergence_window_size', 10)
        )

        # Set interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Set up transform based on transform_type
        initial_transform = None

        if self.transform_type == 'rigid':
            # Start with center aligned transform
            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif self.transform_type == 'affine':
            # Start with center aligned transform
            initial_transform = sitk.CenteredTransformInitializer(
                fixed, moving, sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif self.transform_type == 'deformable':
            # First perform affine registration
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

            # Now set up deformable registration
            # Start with a displacement field transform with zeros
            mesh_size = self.kwargs.get('mesh_size', [10, 10, 10])

            # Create BSpline transform
            transform_domain_mesh_size = [mesh_size[0] - 3, mesh_size[1] - 3, mesh_size[2] - 3]
            transform_domain_origin = fixed.GetOrigin()
            transform_domain_direction = fixed.GetDirection()

            # Get physical dimensions in each direction
            transform_domain_physical_dimensions = [
                fixed.GetSize()[0] * fixed.GetSpacing()[0],
                fixed.GetSize()[1] * fixed.GetSpacing()[1],
                fixed.GetSize()[2] * fixed.GetSpacing()[2]
            ]
            transform_domain_size = fixed.GetSize()

            # Create the BSpline transform
            initial_transform = sitk.BSplineTransformInitializer(
                fixed, transform_domain_mesh_size, order=3
            )

            # Apply the affine transform first
            registration_method.SetInitialTransform(affine_transform, inPlace=False)

            # Use a different optimizer for the BSpline registration
            registration_method.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-5,
                numberOfIterations=self.kwargs.get('num_iterations', 100),
                maximumNumberOfCorrections=5,
                maximumNumberOfFunctionEvaluations=1000,
                costFunctionConvergenceFactor=1e7
            )
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")

        # Set initial transform
        if initial_transform is not None and self.transform_type != 'deformable':
            registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Execute registration
        if self.transform_type != 'deformable':
            transform = registration_method.Execute(fixed, moving)
        else:
            # For deformable, we need to specifically use the BSpline transform
            if self.kwargs.get('use_bspline', True):
                registration_method.SetInitialTransform(initial_transform, inPlace=True)
                transform = registration_method.Execute(fixed, moving)
            else:
                # Alternative: Use displacement field transform
                transform = registration_method.Execute(fixed, moving)

        # Apply transform to moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)

        transformed_moving = resampler.Execute(moving)

        # Convert to numpy array if input was numpy array
        if isinstance(moving_image, np.ndarray):
            transformed_moving_array = sitk.GetArrayFromImage(transformed_moving)
            # Transpose to match input dimensions
            transformed_moving_array = np.transpose(transformed_moving_array,
                                                    list(range(transformed_moving_array.ndim))[::-1])
        else:
            transformed_moving_array = None

        # Return results
        return {
            'transformed_image': transformed_moving,
            'transformed_array': transformed_moving_array,
            'transform': transform,
            'metric_value': registration_method.GetMetricValue(),
            'iterations': registration_method.GetOptimizerIteration(),
            'stop_condition': registration_method.GetOptimizerStopConditionDescription()
        }

    def _load_image(self, image: Union[str, np.ndarray]) -> sitk.Image:
        """
        Load an image from a file or array

        Args:
            image: Image path or array

        Returns:
            SimpleITK image
        """
        if isinstance(image, str):
            return sitk.ReadImage(image)
        elif isinstance(image, np.ndarray):
            # Convert array to SimpleITK image
            # Transpose to match SimpleITK's axis ordering
            image_t = np.transpose(image, list(range(image.ndim))[::-1])
            return sitk.GetImageFromArray(image_t)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")


class ANTsRegistration(RegistrationMethod):
    """
    ANTsPy-based registration

    This class provides registration methods using ANTsPy.
    """

    def __init__(self, transform_type: str = 'rigid', **kwargs):
        """
        Initialize the ANTsPy registration method

        Args:
            transform_type: Type of transformation ('rigid', 'affine', 'deformable')
            **kwargs: Additional parameters for registration
        """
        if not HAS_ANTS:
            raise ImportError("ANTsPy is required for ANTsRegistration")

        super().__init__(**kwargs)
        self.transform_type = transform_type

    def register(self, fixed_image: Union[str, np.ndarray], moving_image: Union[str, np.ndarray],
                 fixed_mask: Optional[Union[str, np.ndarray]] = None,
                 moving_mask: Optional[Union[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Register moving image to fixed image

        Args:
            fixed_image: Fixed/reference image (path or array)
            moving_image: Moving image to be registered (path or array)
            fixed_mask: Optional mask for fixed image (path or array)
            moving_mask: Optional mask for moving image (path or array)

        Returns:
            Dictionary with registration results
        """
        # Load images
        fixed = self._load_image(fixed_image)
        moving = self._load_image(moving_image)

        # Load masks if provided
        fixed_mask_img = None
        if fixed_mask is not None:
            fixed_mask_img = self._load_image(fixed_mask)

        # Set up type of registration
        if self.transform_type == 'rigid':
            ants_transform_type = 'Rigid'
        elif self.transform_type == 'affine':
            ants_transform_type = 'Affine'
        elif self.transform_type == 'deformable':
            ants_transform_type = 'SyN'
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")

        # Set up registration parameters
        reg_params = {
            'typeofTransform': ants_transform_type,
            'initialTransform': None,
            'interpolator': 'linear',
            'verbose': self.kwargs.get('verbose', False)
        }

        # Set number of iterations
        if 'num_iterations' in self.kwargs:
            reg_params['gradientStep'] = self.kwargs.get('learning_rate', 0.1)

            # For deformable, set iterations differently
            if ants_transform_type == 'SyN':
                if isinstance(self.kwargs['num_iterations'], list):
                    reg_params['regIterations'] = self.kwargs['num_iterations']
                else:
                    reg_params['regIterations'] = [self.kwargs['num_iterations'] // 4,
                                                   self.kwargs['num_iterations'] // 2,
                                                   self.kwargs['num_iterations']]
            else:
                if isinstance(self.kwargs['num_iterations'], list):
                    reg_params['regIterations'] = self.kwargs['num_iterations']
                else:
                    reg_params['regIterations'] = [self.kwargs['num_iterations']]

        # Set metric
        if 'metric' in self.kwargs:
            if self.kwargs['metric'] == 'mutual_information':
                reg_params['metric'] = 'mattes'
            elif self.kwargs['metric'] == 'mean_squares':
                reg_params['metric'] = 'meansquares'
            elif self.kwargs['metric'] == 'correlation':
                reg_params['metric'] = 'gcc'
            else:
                reg_params['metric'] = 'mattes'

        # Set up mask parameters
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

        # Convert to numpy array if input was numpy array
        if isinstance(moving_image, np.ndarray):
            transformed_moving_array = transformed_moving.numpy()
        else:
            transformed_moving_array = None

        # Return results
        return {
            'transformed_image': transformed_moving,
            'transformed_array': transformed_moving_array,
            'transform': transform,
            'ANTs_dict': registration
        }

    def _load_image(self, image: Union[str, np.ndarray]) -> 'ants.ANTsImage':
        """
        Load an image from a file or array

        Args:
            image: Image path or array

        Returns:
            ANTs image
        """
        if isinstance(image, str):
            return ants.image_read(image)
        elif isinstance(image, np.ndarray):
            # Convert array to ANTs image
            return ants.from_numpy(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")


def get_registration_method(method: str = 'sitk', transform_type: str = 'rigid', **kwargs) -> RegistrationMethod:
    """
    Get a registration method by name

    Args:
        method: Registration method ('sitk' or 'ants')
        transform_type: Type of transformation
        **kwargs: Additional parameters for registration

    Returns:
        Registration method instance
    """
    if method == 'sitk':
        if not HAS_SITK:
            raise ImportError("SimpleITK is required for SITKRegistration")
        return SITKRegistration(transform_type=transform_type, **kwargs)
    elif method == 'ants':
        if not HAS_ANTS:
            raise ImportError("ANTsPy is required for ANTsRegistration")
        return ANTsRegistration(transform_type=transform_type, **kwargs)
    else:
        raise ValueError(f"Unknown registration method: {method}")


def register_images(fixed_path: str, moving_path: str, output_path: str, transform_type: str = 'rigid',
                    fixed_mask_path: Optional[str] = None, moving_mask_path: Optional[str] = None,
                    method: str = 'sitk', params: Optional[Dict[str, Any]] = None) -> str:
    """
    Register moving image to fixed image and save the result

    Args:
        fixed_path: Path to fixed/reference image
        moving_path: Path to moving image
        output_path: Path to save registered image
        transform_type: Type of transformation ('rigid', 'affine', 'deformable')
        fixed_mask_path: Optional path to mask for fixed image
        moving_mask_path: Optional path to mask for moving image
        method: Registration method ('sitk' or 'ants')
        params: Additional parameters for registration

    Returns:
        Path to registered image
    """
    if params is None:
        params = {}

    # Create registration method
    reg_method = get_registration_method(method, transform_type, **params)

    # Register images
    result = reg_method.register(
        fixed_image=fixed_path,
        moving_image=moving_path,
        fixed_mask=fixed_mask_path,
        moving_mask=moving_mask_path
    )

    # Save registered image
    if method == 'sitk':
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(result['transformed_image'], output_path)
    elif method == 'ants':
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ants.image_write(result['transformed_image'], output_path)

    # Save transform if requested
    if params.get('save_transform', False):
        transform_path = params.get('transform_path')
        if transform_path is None:
            # Create transform path based on output path
            transform_path = os.path.join(
                os.path.dirname(output_path),
                f"{os.path.splitext(os.path.basename(output_path))[0]}_transform.{method}"
            )

        # Save transform
        if method == 'sitk':
            sitk.WriteTransform(result['transform'], transform_path)
        elif method == 'ants':
            # ANTs transformation is a list of transform files
            transform_dir = os.path.dirname(transform_path)
            os.makedirs(transform_dir, exist_ok=True)

            # Simply copy the transform files to the output directory
            import shutil
            for i, t in enumerate(result['transform']):
                ext = os.path.splitext(t)[1]
                dest = os.path.join(transform_dir, f"{os.path.splitext(os.path.basename(transform_path))[0]}_{i}{ext}")
                shutil.copy(t, dest)

    return output_path