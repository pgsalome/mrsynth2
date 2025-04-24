import argparse
import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

from utils.config import parse_args_and_config, Config
from utils.io import ensure_dir, save_json, save_image, save_image_grid
from utils.metrics import compute_psnr, compute_ssim, compute_lpips, MetricsTracker
from data.datasets import create_datasets
from data.dataloader import create_dataloaders
from models import load_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate trained model")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model configuration file")
    parser.add_argument("--output_dir", type=str, default="./evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--dataset_dir", type=str,
                        help="Path to test dataset (overrides config)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int,
                        help="Number of samples to evaluate (if None, use all)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save output images")
    parser.add_argument("--compute_fid", action="store_true",
                        help="Compute FID score (requires pytorch-fid)")
    parser.add_argument("--direction", type=str, choices=["AtoB", "BtoA"],
                        help="Translation direction")

    return parser.parse_args()


def evaluate_model(model, dataloader, output_dir: str,
                   save_images: bool = False, num_samples: Optional[int] = None,
                   compute_fid: bool = False):
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data
        output_dir: Directory to save results
        save_images: Whether to save output images
        num_samples: Number of samples to evaluate
        compute_fid: Whether to compute FID score

    Returns:
        Dictionary with evaluation metrics
    """
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Create directories for saving images
    if save_images:
        real_a_dir = Path(output_dir) / "real_A"
        fake_b_dir = Path(output_dir) / "fake_B"
        real_b_dir = Path(output_dir) / "real_B"

        ensure_dir(real_a_dir)
        ensure_dir(fake_b_dir)
        ensure_dir(real_b_dir)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()

    # Keep track of some images for visualization
    vis_images = []

    # Evaluate model
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Limit number of samples if specified
            if num_samples is not None and i >= num_samples:
                break

            # Move data to device
            device = next(model.parameters()).device
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)

            # Forward pass
            model.set_input(data)
            output = model.test()

            # Get output images
            real_a = output['real_A']
            fake_b = output['fake_B']
            real_b = output['real_B']

            # Calculate metrics
            psnr = compute_psnr(fake_b, real_b)
            ssim = compute_ssim(fake_b, real_b)

            # Update metrics tracker
            metrics_tracker.update('psnr', psnr, real_a.size(0))
            metrics_tracker.update('ssim', ssim, real_a.size(0))

            # Calculate LPIPS if available
            try:
                lpips = compute_lpips(fake_b, real_b)
                metrics_tracker.update('lpips', lpips, real_a.size(0))
            except:
                # LPIPS might not be available
                pass

            # Save images
            if save_images:
                for b in range(real_a.size(0)):
                    idx = i * dataloader.batch_size + b

                    # Save input image
                    save_image(real_a[b], real_a_dir / f"{idx:04d}.png")

                    # Save generated image
                    save_image(fake_b[b], fake_b_dir / f"{idx:04d}.png")

                    # Save target image
                    save_image(real_b[b], real_b_dir / f"{idx:04d}.png")

            # Collect some images for visualization
            if len(vis_images) < 4:  # Collect up to 4 sets
                for b in range(min(real_a.size(0), 1)):  # Take only the first sample from each batch
                    vis_images.append({
                        'real_A': real_a[b].cpu(),
                        'fake_B': fake_b[b].cpu(),
                        'real_B': real_b[b].cpu()
                    })

    # Compute metrics
    metrics = metrics_tracker.compute()

    # Compute FID if requested
    if compute_fid and save_images:
        try:
            from pytorch_fid.fid_score import calculate_fid_given_paths

            # Compute FID
            fid = calculate_fid_given_paths(
                paths=[str(real_b_dir), str(fake_b_dir)],
                batch_size=50,
                device=next(model.parameters()).device,
                dims=2048
            )

            metrics['fid'] = fid
        except ImportError:
            print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")

    # Print metrics
    print("\nEvaluation Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Save metrics
    metrics_path = Path(output_dir) / "metrics.json"
    save_json(metrics, metrics_path)

    # Create visualization grid
    if vis_images:
        # Create grid for each set of images
        for i, images in enumerate(vis_images):
            grid_tensors = [images['real_A'], images['fake_B'], images['real_B']]
            grid_path = Path(output_dir) / f"samples_{i}.png"
            save_image_grid(grid_tensors, grid_path, nrow=3)

        # Create combined grid
        all_tensors = []
        for images in vis_images:
            all_tensors.extend([images['real_A'], images['fake_B'], images['real_B']])

        grid_path = Path(output_dir) / "samples_grid.png"
        save_image_grid(all_tensors, grid_path, nrow=3)

    return metrics


def main():
    args = parse_args()

    # Load configuration
    config = Config.from_file(args.config)

    # Override configuration with command line arguments
    if args.dataset_dir:
        config["data.dataset_dir"] = args.dataset_dir

    if args.batch_size:
        config["data.batch_size"] = args.batch_size

    if args.direction:
        config["data.direction"] = args.direction

    # Set phase to test
    config["data.phase"] = "test"

    # Create dataset and dataloader
    print("Creating test dataset...")
    datasets = create_datasets(config.to_dict())

    if "test" not in datasets:
        raise ValueError("No test dataset available")

    dataloaders = create_dataloaders({"test": datasets["test"]}, config.to_dict())
    test_dataloader = dataloaders["test"]

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

    # Evaluate model
    print(f"Evaluating {model_type} model on {len(datasets['test'])} test samples...")
    evaluate_model(
        model=model,
        dataloader=test_dataloader,
        output_dir=output_dir,
        save_images=args.save_images,
        num_samples=args.num_samples,
        compute_fid=args.compute_fid
    )

    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()