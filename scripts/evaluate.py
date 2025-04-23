import json
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm

from src.utils.io import ensure_dir, save_image, save_image_grid
from src.utils.config import read_json_config
from src.utils.metrics import compute_psnr, compute_ssim, compute_lpips
from src.data_loader import create_datasets, create_dataloaders
from models.model_factory import load_model_from_checkpoint


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate image-to-image translation model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model and configuration")
    parser.add_argument("--dataset_dir", type=str, help="Directory containing the dataset (overrides config)")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save evaluation results (default: model_dir/evaluation)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num_test", type=int, help="Number of test images to evaluate")
    parser.add_argument("--compute_fid", action="store_true", help="Compute FID score (slower)")
    parser.add_argument("--direction", type=str, choices=["AtoB", "BtoA"],
                        help="Direction for evaluation (only for CycleGAN)")
    return parser.parse_args()


def load_model_from_dir(model_dir: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model and configuration from directory.

    Args:
        model_dir: Directory containing the model and configuration
        device: Device to load the model on

    Returns:
        Tuple of (model, config)
    """
    model_dir = Path(model_dir)

    # Load configuration
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Configuration file not found: {config_path}")

    config = read_json_config(str(config_path))

    # Extract model type from config
    model_type = config["model"]["name"]
    print(f"Detected model type: {model_type}")

    # Load model using factory
    model = load_model_from_checkpoint(
        model_dir=str(model_dir),
        epoch="best",
        config=config,
        device=device
    )

    # Set model to evaluation mode
    model.eval()

    return model, config


def evaluate_model(model_dir: str, dataset_dir: Optional[str] = None, output_dir: Optional[str] = None,
                   batch_size: int = 1, num_test: Optional[int] = None, compute_fid: bool = False,
                   direction: Optional[str] = None):
    """
    Evaluate a trained model.

    Args:
        model_dir: Directory containing the model and configuration
        dataset_dir: Optional directory containing the dataset (overrides config)
        output_dir: Optional directory to save evaluation results
        batch_size: Batch size for evaluation
        num_test: Number of test images to evaluate
        compute_fid: Whether to compute FID score
        direction: Direction for evaluation (only for CycleGAN)
    """
    # Set up output directory
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir / "evaluation"
    else:
        output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and config
    model, config = load_model_from_dir(model_dir, device)
    model_type = config["model"]["name"]

    # Override config with command line arguments
    if dataset_dir:
        config["data"]["dataset_dir"] = dataset_dir

    if batch_size:
        config["data"]["batch_size"] = batch_size

    if direction:
        config["data"]["direction"] = direction

    # Create test dataset and dataloader
    print("Creating test dataset...")
    datasets = create_datasets(config)

    if "test" not in datasets:
        print("Warning: No test dataset found. Using validation dataset.")
        if "val" in datasets:
            datasets["test"] = datasets["val"]
        else:
            raise ValueError("No test or validation dataset found.")

    # Limit number of test samples if specified
    if num_test and num_test > 0:
        # Create subset of test dataset
        from torch.utils.data import Subset
        indices = list(range(min(num_test, len(datasets["test"]))))
        datasets["test"] = Subset(datasets["test"], indices)
        print(f"Using {len(indices)} test samples")

    # Create dataloader
    dataloaders = create_dataloaders({"test": datasets["test"]}, config)
    test_dataloader = dataloaders["test"]

    print(f"Evaluating {model_type} model on {len(datasets['test'])} test samples...")

    # Prepare for evaluation
    metrics = {
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0
    }

    # Initialize LPIPS model
    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
    except ImportError:
        print("LPIPS package not available. Install with: pip install lpips")

    # Create directories for saving images
    real_A_dir = output_dir / "real_A"
    fake_B_dir = output_dir / "fake_B"
    real_B_dir = output_dir / "real_B"

    for dir_path in [real_A_dir, fake_B_dir, real_B_dir]:
        ensure_dir(dir_path)

    # Generate and evaluate
    model.eval()

    # Collect images for visualization and FID calculation
    all_real_A = []
    all_fake_B = []
    all_real_B = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            # Move data to device
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)

            # Forward pass
            model.set_input(data)
            output = model.test()

            # Get output images
            real_A = output['real_A']
            fake_B = output['fake_B']
            real_B = output['real_B']

            # Calculate metrics
            metrics['psnr'] += compute_psnr(fake_B, real_B)
            metrics['ssim'] += compute_ssim(fake_B, real_B)

            if lpips_model is not None:
                metrics['lpips'] += compute_lpips(fake_B, real_B, lpips_model)

            # Save images
            for b in range(real_A.size(0)):
                img_idx = i * test_dataloader.batch_size + b

                # Save input image
                save_image(real_A[b], real_A_dir / f"{img_idx:04d}.png")

                # Save generated image
                save_image(fake_B[b], fake_B_dir / f"{img_idx:04d}.png")

                # Save target image
                save_image(real_B[b], real_B_dir / f"{img_idx:04d}.png")

                # Collect for visualization
                if len(all_real_A) < 16:  # Limit to 16 images for visualization
                    all_real_A.append(real_A[b:b + 1])
                    all_fake_B.append(fake_B[b:b + 1])
                    all_real_B.append(real_B[b:b + 1])

    # Average metrics
    num_test_samples = len(datasets["test"])
    for key in metrics:
        metrics[key] /= num_test_samples

    # Compute FID score if requested
    if compute_fid:
        print("Computing FID score...")
        try:
            fid_score = compute_fid(str(real_B_dir), str(fake_B_dir))
            metrics['fid'] = fid_score
            print(f"FID score: {fid_score:.4f}")
        except Exception as e:
            print(f"Error computing FID score: {e}")

    # Print metrics
    print(f"\nEvaluation metrics for {model_type} model:")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.4f}")

    # Save metrics to file
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Create visualization of results
    if all_real_A and all_fake_B and all_real_B:
        print("Creating visualization...")

        # Combine input, generated, and target images into a grid
        vis_images = []
        for i in range(min(8, len(all_real_A))):
            vis_images.extend([all_real_A[i], all_fake_B[i], all_real_B[i]])

        # Save grid
        grid_path = output_dir / "results_grid.png"
        save_image_grid(vis_images, grid_path, nrow=3)

        print(f"Visualization saved to {grid_path}")

    print(f"\nEvaluation completed. Results saved to {output_dir}")
    return metrics


def main():
    """Main function"""
    args = parse_args()
    metrics = evaluate_model(
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_test=args.num_test,
        compute_fid=args.compute_fid,
        direction=args.direction
    )


if __name__ == "__main__":
    main()