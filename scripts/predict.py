import argparse
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import numpy as np

from utils.config import parse_args_and_config, Config
from utils.io import ensure_dir, save_image, save_image_grid, load_image
from utils.metrics import compute_psnr, compute_ssim, compute_lpips
from models import load_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with trained model")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model configuration file")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image or directory")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--target", type=str,
                        help="Path to target image or directory (for evaluation)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--direction", type=str, choices=["AtoB", "BtoA"],
                        help="Translation direction")

    return parser.parse_args()


def process_single_image(model, input_path: str, output_dir: str,
                         target_path: Optional[str] = None, direction: str = "AtoB"):
    """
    Process a single image.

    Args:
        model: Model to use
        input_path: Path to input image
        output_dir: Directory to save results
        target_path: Path to target image (optional)
        direction: Translation direction

    Returns:
        Dictionary with results
    """
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Load input image
    input_tensor = load_image(input_path)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Load target image if provided
    target_tensor = None
    if target_path:
        target_tensor = load_image(target_path)
        target_tensor = target_tensor.unsqueeze(0)  # Add batch dimension

    # Move tensors to device
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    if target_tensor is not None:
        target_tensor = target_tensor.to(device)

    # Create input dictionary
    input_dict = {'A': input_tensor}
    if target_tensor is not None:
        input_dict['B'] = target_tensor

    # Run inference
    with torch.no_grad():
        # Set model input
        model.set_input(input_dict)

        # Forward pass
        output = model.test()

    # Get output image
    output_tensor = output['fake_B']

    # Save results
    input_name = Path(input_path).stem

    # Save input image
    input_path = Path(output_dir) / f"{input_name}_input.png"
    save_image(input_tensor[0], input_path)

    # Save output image
    output_path = Path(output_dir) / f"{input_name}_output.png"
    save_image(output_tensor[0], output_path)

    # Save target image if provided
    target_path_out = None
    if target_tensor is not None:
        target_path_out = Path(output_dir) / f"{input_name}_target.png"
        save_image(target_tensor[0], target_path_out)

    # Calculate metrics if target is provided
    metrics = {}
    if target_tensor is not None:
        metrics['psnr'] = compute_psnr(output_tensor, target_tensor)
        metrics['ssim'] = compute_ssim(output_tensor, target_tensor)
        try:
            metrics['lpips'] = compute_lpips(output_tensor, target_tensor)
        except:
            # LPIPS might not be available
            pass

    # Print metrics
    if metrics:
        print(f"Metrics for {input_name}:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    # Create side-by-side visualization
    if target_tensor is not None:
        # Combine input, output, and target
        grid_tensors = [input_tensor[0], output_tensor[0], target_tensor[0]]
        grid_path = Path(output_dir) / f"{input_name}_comparison.png"
        save_image_grid(grid_tensors, grid_path, nrow=3)
    else:
        # Combine input and output
        grid_tensors = [input_tensor[0], output_tensor[0]]
        grid_path = Path(output_dir) / f"{input_name}_comparison.png"
        save_image_grid(grid_tensors, grid_path, nrow=2)

    return {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'target_path': str(target_path_out) if target_path_out else None,
        'metrics': metrics
    }


def process_directory(model, input_dir: str, output_dir: str,
                      target_dir: Optional[str] = None, direction: str = "AtoB"):
    """
    Process all images in a directory.

    Args:
        model: Model to use
        input_dir: Directory with input images
        output_dir: Directory to save results
        target_dir: Directory with target images (optional)
        direction: Translation direction

    Returns:
        List of result dictionaries
    """
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Find input images
    input_dir = Path(input_dir)
    input_paths = list(input_dir.glob('*.png'))
    input_paths.extend(input_dir.glob('*.jpg'))

    # Find target images if provided
    target_paths = None
    if target_dir:
        target_dir = Path(target_dir)
        target_paths = {}
        for path in target_dir.glob('*.png'):
            target_paths[path.stem] = path
        for path in target_dir.glob('*.jpg'):
            target_paths[path.stem] = path

    # Process each image
    results = []
    for input_path in input_paths:
        # Find matching target image if available
        target_path = None
        if target_paths and input_path.stem in target_paths:
            target_path = target_paths[input_path.stem]

        # Process image
        result = process_single_image(
            model=model,
            input_path=str(input_path),
            output_dir=output_dir,
            target_path=str(target_path) if target_path else None,
            direction=direction
        )

        results.append(result)

    # Calculate average metrics
    if results and 'metrics' in results[0] and results[0]['metrics']:
        metrics = {}
        for metric in results[0]['metrics'].keys():
            values = [r['metrics'][metric] for r in results if metric in r['metrics']]
            metrics[metric] = sum(values) / len(values)

        print("\nAverage metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    return results


def main():
    args = parse_args()

    # Load configuration
    config = Config.from_file(args.config)

    # Override configuration with command line arguments
    if args.direction:
        config["data.direction"] = args.direction

    if args.batch_size:
        config["data.batch_size"] = args.batch_size

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model_type = config["model.name"]
    model = load_model(model_type, args.model_path, config.to_dict(), device)
    model.eval()

    # Create output directory
    output_dir = args.output_dir
    ensure_dir(output_dir)

    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single image
        print(f"Processing single image: {input_path}")
        process_single_image(
            model=model,
            input_path=str(input_path),
            output_dir=output_dir,
            target_path=args.target,
            direction=config.get("data.direction", "AtoB")
        )
    elif input_path.is_dir():
        # Process directory
        print(f"Processing directory: {input_path}")
        process_directory(
            model=model,
            input_dir=str(input_path),
            output_dir=output_dir,
            target_dir=args.target,
            direction=config.get("data.direction", "AtoB")
        )
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()