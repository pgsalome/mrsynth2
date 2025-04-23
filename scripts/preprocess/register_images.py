#!/usr/bin/env python
"""
Register MRI images

This script registers MRI images to a reference/fixed image or template.
It supports different registration methods and transformations, including:
- Rigid (translation and rotation)
- Affine (translation, rotation, scaling, and shearing)
- Deformable/non-rigid registration

Both SimpleITK and ANTs registration frameworks are supported if available.
"""

import os
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Try to import SimpleITK first
try:
    import SimpleITK as sitk

    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    logging.warning("SimpleITK not available. Some registration methods may be limited.")

# Try to import ANTs through ANTsPy if available
try:
    import ants

    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False
    logging.warning("ANTsPy not available. Some registration methods may be limited.")

# Try to import our own registration module
try:
    from src.preprocessing.registration import register_images as reg_module

    HAS_REG_MODULE = True
except ImportError:
    HAS_REG_MODULE = False
    logging.warning("Custom registration module not available.")

# Check if any registration method is available
if not any([HAS_SITK, HAS_ANTS, HAS_REG_MODULE]):
    raise ImportError("No registration method available. Please install SimpleITK, ANTsPy, or provide custom module.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_image(file_path, framework='sitk'):
    """
    Load an image using the specified framework

    Args:
        file_path: Path to image file
        framework: Framework to use for loading ('sitk' or 'ants')

    Returns:
        Loaded image in the specified framework
    """
    if framework == 'sitk' and HAS_SITK:
        return sitk.ReadImage(file_path)
    elif framework == 'ants' and HAS_ANTS:
        return ants.image_read(file_path)
    elif HAS_SITK:
        return sitk.ReadImage(file_path)
    elif HAS_ANTS:
        return ants.image_read(file_path)
    else:
        raise ImportError("No image loading framework available")


def save_image(image, file_path, framework='sitk'):
    """
    Save an image using the specified framework

    Args:
        image: Image to save
        file_path: Path to save the image
        framework: Framework to use for saving ('sitk' or 'ants')
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if framework == 'sitk' and HAS_SITK:
        sitk.WriteImage(image, file_path)
    elif framework == 'ants' and HAS_ANTS:
        ants.image_write(image, file_path)
    elif HAS_SITK:
        sitk.WriteImage(image, file_path)
    elif HAS_ANTS:
        ants.image_write(image, file_path)
    else:
        raise ImportError("No image saving framework available")


def register_sitk(fixed_image, moving_image, transform_type, fixed_mask=None, moving_mask=None, params=None):
    """
    Register moving image to fixed image using SimpleITK

    Args:
        fixed_image: Fixed/reference image
        moving_image: Moving image to be registered
        transform_type: Type of transformation ('rigid', 'affine', 'deformable')
        fixed_mask: Optional mask for fixed image
        moving_mask: Optional mask for moving image
        params: Additional parameters for registration

    Returns:
        Tuple of (transformed moving image, transformation)
    """
    # Set default parameters
    if params is None:
        params = {}

    # Initialize registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Set up metric
    metric = params.get('metric', 'mutual_information')
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
    if fixed_mask is not None:
        registration_method.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        registration_method.SetMetricMovingMask(moving_mask)

    # Set optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=params.get('learning_rate', 1.0),
        numberOfIterations=params.get('num_iterations', 100),
        convergenceMinimumValue=params.get('convergence_min_value', 1e-6),
        convergenceWindowSize=params.get('convergence_window_size', 10)
    )

    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Set up transform based on transform_type
    initial_transform = None

    if transform_type == 'rigid':
        # Start with center aligned transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif transform_type == 'affine':
        # Start with center aligned transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif transform_type == 'deformable':
        # First perform affine registration
        affine_registration = sitk.ImageRegistrationMethod()
        affine_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        affine_registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
        affine_registration.SetInterpolator(sitk.sitkLinear)

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        affine_registration.SetInitialTransform(initial_transform, inPlace=False)
        affine_transform = affine_registration.Execute(fixed_image, moving_image)

        # Now set up deformable registration
        # Start with a displacement field transform with zeros
        mesh_size = params.get('mesh_size', [10, 10, 10])

        # Create BSpline transform
        transform_domain_mesh_size = [mesh_size[0] - 3, mesh_size[1] - 3, mesh_size[2] - 3]
        transform_domain_origin = fixed_image.GetOrigin()
        transform_domain_direction = fixed_image.GetDirection()

        # Get physical dimensions in each direction
        transform_domain_physical_dimensions = [
            fixed_image.GetSize()[0] * fixed_image.GetSpacing()[0],
            fixed_image.GetSize()[1] * fixed_image.GetSpacing()[1],
            fixed_image.GetSize()[2] * fixed_image.GetSpacing()[2]
        ]
        transform_domain_size = fixed_image.GetSize()

        # Create the BSpline transform
        initial_transform = sitk.BSplineTransformInitializer(
            fixed_image, transform_domain_mesh_size, order=3
        )

        # Apply the affine transform first
        registration_method.SetInitialTransform(affine_transform, inPlace=False)

        # Use a different optimizer for the BSpline registration
        registration_method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=params.get('num_iterations', 100),
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e7
        )
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    # Set initial transform
    if initial_transform is not None and transform_type != 'deformable':
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Execute registration
    if transform_type != 'deformable':
        transform = registration_method.Execute(fixed_image, moving_image)
    else:
        # For deformable, we need to specifically use the BSpline transform
        if params.get('use_bspline', True):
            registration_method.SetInitialTransform(initial_transform, inPlace=True)
            transform = registration_method.Execute(fixed_image, moving_image)
        else:
            # Alternative: Use displacement field transform
            transform = registration_method.Execute(fixed_image, moving_image)

    # Apply transform to moving image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    transformed_moving_image = resampler.Execute(moving_image)

    return transformed_moving_image, transform


def register_ants(fixed_image, moving_image, transform_type, fixed_mask=None, moving_mask=None, params=None):
    """
    Register moving image to fixed image using ANTs (ANTsPy)

    Args:
        fixed_image: Fixed/reference image
        moving_image: Moving image to be registered
        transform_type: Type of transformation ('rigid', 'affine', 'deformable')
        fixed_mask: Optional mask for fixed image
        moving_mask: Optional mask for moving image
        params: Additional parameters for registration

    Returns:
        Tuple of (transformed moving image, transformation)
    """
    # Set default parameters
    if params is None:
        params = {}

    # Set up type of registration
    if transform_type == 'rigid':
        transform_type = 'Rigid'
    elif transform_type == 'affine':
        transform_type = 'Affine'
    elif transform_type == 'deformable':
        transform_type = 'SyN'
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    # Set up registration parameters
    reg_params = {
        'typeofTransform': transform_type,
        'initialTransform': None,
        'interpolator': 'linear',
        'verbose': params.get('verbose', False)
    }

    # Set number of iterations
    if 'num_iterations' in params:
        reg_params['gradientStep'] = params.get('learning_rate', 0.1)

        # For deformable, set iterations differently
        if transform_type == 'SyN':
            if isinstance(params['num_iterations'], list):
                reg_params['regIterations'] = params['num_iterations']
            else:
                reg_params['regIterations'] = [params['num_iterations'] // 4,
                                               params['num_iterations'] // 2,
                                               params['num_iterations']]
        else:
            if isinstance(params['num_iterations'], list):
                reg_params['regIterations'] = params['num_iterations']
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
        else:
            reg_params['metric'] = 'mattes'

    # Set up mask parameters
    if fixed_mask is not None:
        reg_params['mask'] = fixed_mask

    # Perform registration
    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        **reg_params
    )

    # Extract results
    transformed_moving_image = registration['warpedmovout']
    transform = registration['fwdtransforms']

    return transformed_moving_image, transform


def register_images(fixed_path, moving_path, output_path, transform_type='rigid',
                    fixed_mask_path=None, moving_mask_path=None,
                    framework='auto', params=None):
    """
    Register moving image to fixed image and save the result

    Args:
        fixed_path: Path to fixed/reference image
        moving_path: Path to moving image
        output_path: Path to save registered image
        transform_type: Type of transformation ('rigid', 'affine', 'deformable')
        fixed_mask_path: Optional path to mask for fixed image
        moving_mask_path: Optional path to mask for moving image
        framework: Framework to use ('sitk', 'ants', or 'auto')
        params: Additional parameters for registration

    Returns:
        Path to registered image
    """
    # Determine which framework to use
    if framework == 'auto':
        if HAS_REG_MODULE:
            # Use our custom module
            return reg_module.register_images(
                fixed_path, moving_path, output_path, transform_type,
                fixed_mask_path, moving_mask_path, params
            )
        elif HAS_ANTS:
            framework = 'ants'
        elif HAS_SITK:
            framework = 'sitk'
        else:
            raise ImportError("No registration framework available")

    # Load images
    fixed_image = load_image(fixed_path, framework)
    moving_image = load_image(moving_path, framework)

    # Load masks if provided
    fixed_mask = None
    moving_mask = None

    if fixed_mask_path and os.path.exists(fixed_mask_path):
        fixed_mask = load_image(fixed_mask_path, framework)

    if moving_mask_path and os.path.exists(moving_mask_path):
        moving_mask = load_image(moving_mask_path, framework)

    # Perform registration
    if framework == 'sitk':
        transformed_moving_image, transform = register_sitk(
            fixed_image, moving_image, transform_type, fixed_mask, moving_mask, params
        )
    elif framework == 'ants':
        transformed_moving_image, transform = register_ants(
            fixed_image, moving_image, transform_type, fixed_mask, moving_mask, params
        )
    else:
        raise ValueError(f"Unknown framework: {framework}")

    # Save registered image
    save_image(transformed_moving_image, output_path, framework)

    # Save transform if requested
    if params and params.get('save_transform', False):
        transform_path = params.get('transform_path')
        if transform_path is None:
            # Create transform path based on output path
            transform_path = os.path.join(
                os.path.dirname(output_path),
                f"{Path(output_path).stem}_transform.{framework}"
            )

        # Save transform
        if framework == 'sitk':
            sitk.WriteTransform(transform, transform_path)
        elif framework == 'ants':
            # ANTs transformation is a list of transform files
            transform_dir = os.path.dirname(transform_path)
            os.makedirs(transform_dir, exist_ok=True)

            # Simply copy the transform files to the output directory
            import shutil
            for i, t in enumerate(transform):
                ext = os.path.splitext(t)[1]
                dest = os.path.join(transform_dir, f"{Path(transform_path).stem}_{i}{ext}")
                shutil.copy(t, dest)

    return output_path


def process_registration(reg_info):
    """Process a single registration task"""
    try:
        fixed_path = reg_info['fixed_path']
        moving_path = reg_info['moving_path']
        output_path = reg_info['output_path']
        transform_type = reg_info.get('transform_type', 'rigid')
        fixed_mask_path = reg_info.get('fixed_mask_path')
        moving_mask_path = reg_info.get('moving_mask_path')
        framework = reg_info.get('framework', 'auto')
        params = reg_info.get('params', {})

        return register_images(
            fixed_path, moving_path, output_path, transform_type,
            fixed_mask_path, moving_mask_path, framework, params
        )
    except Exception as e:
        logger.error(f"Error registering {moving_path} to {fixed_path}: {e}")
        return None


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Register MRI images")

    parser.add_argument("--fixed_dir", type=str,
                        help="Directory containing fixed/reference images")
    parser.add_argument("--moving_dir", type=str,
                        help="Directory containing moving images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for registered images")
    parser.add_argument("--registration_list", type=str,
                        help="JSON file with list of registrations to perform")
    parser.add_argument("--transform_type", type=str, default="rigid",
                        choices=["rigid", "affine", "deformable"],
                        help="Type of transformation to apply")
    parser.add_argument("--fixed_mask_dir", type=str,
                        help="Directory containing masks for fixed images")
    parser.add_argument("--moving_mask_dir", type=str,
                        help="Directory containing masks for moving images")
    parser.add_argument("--framework", type=str, default="auto",
                        choices=["auto", "sitk", "ants"],
                        help="Registration framework to use")
    parser.add_argument("--metric", type=str, default="mutual_information",
                        choices=["mutual_information", "mean_squares", "correlation"],
                        help="Metric for registration")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate for optimizer")
    parser.add_argument("--num_iterations", type=int, default=100,
                        help="Number of iterations for optimizer")
    parser.add_argument("--save_transform", action="store_true",
                        help="Save transformation parameters")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of registrations to perform
    if args.registration_list:
        # Load from JSON file
        with open(args.registration_list, 'r') as f:
            registration_infos = json.load(f)
    elif args.fixed_dir and args.moving_dir:
        # Auto-generate registration pairs
        registration_infos = []

        # Get all NIfTI files in the fixed and moving directories
        fixed_files = []
        for root, _, files in os.walk(args.fixed_dir):
            for f in files:
                if f.endswith(('.nii', '.nii.gz')):
                    fixed_files.append(os.path.join(root, f))

        moving_files = []
        for root, _, files in os.walk(args.moving_dir):
            for f in files:
                if f.endswith(('.nii', '.nii.gz')):
                    moving_files.append(os.path.join(root, f))

        # Create registration pairs based on filenames
        # This assumes files with the same stem (ignoring extensions) should be paired
        fixed_dict = {Path(f).stem.split('.')[0]: f for f in fixed_files}
        moving_dict = {Path(f).stem.split('.')[0]: f for f in moving_files}

        # Find common subjects/scans
        common_keys = set(fixed_dict.keys()).intersection(set(moving_dict.keys()))

        for key in common_keys:
            fixed_path = fixed_dict[key]
            moving_path = moving_dict[key]

            # Determine output path
            rel_path = os.path.relpath(moving_path, args.moving_dir)
            output_path = os.path.join(args.output_dir, rel_path)

            # Determine mask paths if mask directories provided
            fixed_mask_path = None
            if args.fixed_mask_dir:
                rel_fixed_path = os.path.relpath(fixed_path, args.fixed_dir)
                fixed_mask_path = os.path.join(args.fixed_mask_dir, rel_fixed_path)
                if not os.path.exists(fixed_mask_path):
                    # Try with _mask suffix
                    mask_name = Path(rel_fixed_path).stem + '_mask.nii.gz'
                    fixed_mask_path = os.path.join(
                        args.fixed_mask_dir,
                        os.path.dirname(rel_fixed_path),
                        mask_name
                    )
                    if not os.path.exists(fixed_mask_path):
                        fixed_mask_path = None

            moving_mask_path = None
            if args.moving_mask_dir:
                rel_moving_path = os.path.relpath(moving_path, args.moving_dir)
                moving_mask_path = os.path.join(args.moving_mask_dir, rel_moving_path)
                if not os.path.exists(moving_mask_path):
                    # Try with _mask suffix
                    mask_name = Path(rel_moving_path).stem + '_mask.nii.gz'
                    moving_mask_path = os.path.join(
                        args.moving_mask_dir,
                        os.path.dirname(rel_moving_path),
                        mask_name
                    )
                    if not os.path.exists(moving_mask_path):
                        moving_mask_path = None

            # Create registration info
            registration_infos.append({
                'fixed_path': fixed_path,
                'moving_path': moving_path,
                'output_path': output_path,
                'transform_type': args.transform_type,
                'fixed_mask_path': fixed_mask_path,
                'moving_mask_path': moving_mask_path,
                'framework': args.framework,
                'params': {
                    'metric': args.metric,
                    'learning_rate': args.learning_rate,
                    'num_iterations': args.num_iterations,
                    'save_transform': args.save_transform
                }
            })
    else:
        logger.error("Either --registration_list or both --fixed_dir and --moving_dir must be provided.")
        exit(1)

    # Process registrations
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            results = list(tqdm(
                executor.map(process_registration, registration_infos),
                total=len(registration_infos),
                desc="Registering images"
            ))
    else:
        results = []
        for reg_info in tqdm(registration_infos, desc="Registering images"):
            results.append(process_registration(reg_info))

    # Count successes
    successful = [r for r in results if r is not None]
    logger.info(
        f"Registration completed. {len(successful)}/{len(registration_infos)} registrations processed successfully.")