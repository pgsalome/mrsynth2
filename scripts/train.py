import os
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import wandb for experiment tracking
import wandb

# Import tensorboard for local logging
from torch.utils.tensorboard import SummaryWriter

# Import utilities
from src.utils.io import ensure_dir, save_model, save_json, save_image_grid
from src.utils.config import load_model_config
from src.utils.metrics import compute_psnr, compute_ssim, compute_lpips, EarlyStopping
from src.data_loader import create_datasets, create_dataloaders
from models.model_factory import get_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image-to-image translation models')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['cyclegan', 'pix2pix.json', 'diffusion.json', 'latent_diffusion', 'vae'],
                        help='Model type to train')
    parser.add_argument('--base_config', type=str, default='config/base.json',
                        help='Path to base config file')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs to use (comma-separated)')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--output_dir', type=str, help='Directory to save outputs')
    return parser.parse_args()


def setup_environment(config: Dict[str, Any]) -> torch.device:
    """
    Set up training environment.

    Args:
        config: Configuration dictionary

    Returns:
        Device to use for training
    """
    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set up device
    gpu_ids = config.get("gpu_ids", "0")
    gpu_ids = [int(id) for id in gpu_ids.split(',') if id.strip()]

    if torch.cuda.is_available() and len(gpu_ids) > 0:
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"Using GPU: {gpu_ids}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Set up output directories
    for dir_path in [config["logging"]["save_model_dir"], config["logging"]["log_dir"]]:
        ensure_dir(dir_path)

    return device


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        epoch: int,
        config: Dict[str, Any],
        writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        device: Device to train on
        epoch: Current epoch number
        config: Training configuration
        writer: TensorBoard SummaryWriter

    Returns:
        Dictionary with training metrics
    """
    model.train()

    # Progress tracking
    log_interval = config["logging"]["log_interval"]
    total_batches = len(dataloader)

    # Training metrics
    metrics = {}
    running_losses = {}

    # Time tracking
    start_time = time.time()
    batch_time = time.time()

    for batch_idx, data in enumerate(dataloader):
        # Move data to device
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)

        # Forward and backward pass
        model.set_input(data)
        model.optimize_parameters()

        # Get current losses
        losses = model.get_current_losses()

        # Update running losses
        for loss_name, loss_value in losses.items():
            if loss_name not in running_losses:
                running_losses[loss_name] = 0.0
            running_losses[loss_name] += loss_value

        # Log progress
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            elapsed_time = time.time() - batch_time
            batch_time = time.time()

            # Print progress
            progress = (batch_idx + 1) / total_batches * 100
            print(f"Epoch {epoch:3d} [{batch_idx + 1:4d}/{total_batches:4d}] "
                  f"({progress:5.1f}%) - "
                  f"Time: {elapsed_time:.2f}s")

            # Print losses
            losses_str = " | ".join([f"{name}: {value:.4f}" for name, value in losses.items()])
            print(f"  Losses: {losses_str}")

            # Log to wandb if enabled
            if config["logging"]["wandb"]["enabled"]:
                # Add batch info to losses
                losses_with_step = {
                    f"train/{name}": value for name, value in losses.items()
                }
                losses_with_step["epoch"] = epoch
                losses_with_step["batch"] = batch_idx

                wandb.log(losses_with_step)

            # Log to tensorboard if enabled
            if writer is not None:
                global_step = (epoch - 1) * total_batches + batch_idx
                for name, value in losses.items():
                    writer.add_scalar(f"train/{name}", value, global_step)

    # Calculate average losses
    for loss_name, loss_value in running_losses.items():
        metrics[loss_name] = loss_value / total_batches

    # Log average losses for the epoch
    print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")
    print(f"  Average losses: {' | '.join([f'{name}: {value:.4f}' for name, value in metrics.items()])}")

    return metrics


def validate(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        epoch: int,
        config: Dict[str, Any],
        writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """
    Validate model on validation set.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        device: Device to validate on
        epoch: Current epoch number
        config: Validation configuration
        writer: TensorBoard SummaryWriter

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    # Validation metrics
    val_metrics = {
        'psnr': 0.0,
        'ssim': 0.0,
        'lpips': 0.0
    }

    # Initialize LPIPS model if needed
    lpips_model = None
    if 'lpips' in val_metrics:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').to(device)
        except ImportError:
            print("LPIPS not available. Install with: pip install lpips")

    # Collect some generated images for visualization
    num_vis_images = min(8, config["data"]["batch_size"])
    vis_images = {
        'real_A': [],
        'fake_B': [],
        'real_B': []
    }

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # Move data to device
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)

            # Forward pass
            model.set_input(data)
            output = model.test()

            # Calculate metrics
            fake_B = output['fake_B']
            real_B = output['real_B']

            # PSNR
            val_metrics['psnr'] += compute_psnr(fake_B, real_B)

            # SSIM
            val_metrics['ssim'] += compute_ssim(fake_B, real_B)

            # LPIPS
            if lpips_model is not None:
                val_metrics['lpips'] += compute_lpips(fake_B, real_B, lpips_model)

            # Collect some images for visualization
            if batch_idx == 0:
                for key in vis_images.keys():
                    if key in output:
                        for i in range(min(num_vis_images, output[key].size(0))):
                            vis_images[key].append(output[key][i:i + 1])

    # Average metrics
    for key in val_metrics:
        val_metrics[key] /= len(dataloader)

    # Log metrics
    print(f"Validation Epoch {epoch}:")
    print(f"  PSNR: {val_metrics['psnr']:.4f} | SSIM: {val_metrics['ssim']:.4f} | LPIPS: {val_metrics['lpips']:.4f}")

    # Log to wandb if enabled
    if config["logging"]["wandb"]["enabled"]:
        # Prepare metrics for wandb
        wandb_metrics = {
            f"val/{key}": value for key, value in val_metrics.items()
        }
        wandb_metrics["epoch"] = epoch

        # Log metrics
        wandb.log(wandb_metrics)

        # Log images
        if config["logging"]["image_log_freq"] > 0 and epoch % config["logging"]["image_log_freq"] == 0:
            # Log A->B direction
            wandb_images = []
            for i in range(len(vis_images['real_A'])):
                wandb_images.append(wandb.Image(
                    vis_images['real_A'][i],
                    caption=f"Input {i}"
                ))
                wandb_images.append(wandb.Image(
                    vis_images['fake_B'][i],
                    caption=f"Generated {i}"
                ))
                wandb_images.append(wandb.Image(
                    vis_images['real_B'][i],
                    caption=f"Target {i}"
                ))

            wandb.log({f"val/images_epoch_{epoch}": wandb_images})

    # Log to tensorboard if enabled
    if writer is not None:
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        # Log images
        if config["logging"]["image_log_freq"] > 0 and epoch % config["logging"]["image_log_freq"] == 0:
            # Create image grids for each type
            for key, images in vis_images.items():
                if images:
                    writer.add_images(f"val/{key}", torch.cat(images, 0), epoch)

    # Save some validation images
    if config["logging"]["image_log_freq"] > 0 and epoch % config["logging"]["image_log_freq"] == 0:
        # Create directory for images
        image_dir = Path(config["logging"]["log_dir"]) / f"images_epoch_{epoch}"
        ensure_dir(image_dir)

        # Save image grid
        grid_path = image_dir / "image_grid.png"

        # Combine images: [real_A, fake_B, real_B] for each sample
        grid_images = []
        for i in range(len(vis_images['real_A'])):
            grid_images.extend([
                vis_images['real_A'][i],
                vis_images['fake_B'][i],
                vis_images['real_B'][i]
            ])

        save_image_grid(grid_images, grid_path, nrow=3)

    return val_metrics


def train(config: Dict[str, Any], model_type: str) -> Dict[str, float]:
    """
    Train model according to configuration.

    Args:
        config: Training configuration
        model_type: Type of model to train

    Returns:
        Dictionary with final metrics
    """
    # Set up training environment
    device = setup_environment(config)

    # Create unique run name for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = model_type
    if "name" in config and config["name"]:
        run_name = f"{run_name}_{config['name']}"
    run_name = f"{run_name}_{timestamp}"

    # Update config with run name
    config["run_name"] = run_name

    # Set up wandb if enabled
    if config["logging"]["wandb"]["enabled"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(
            project=wandb_config.get("project", "mrsynth2"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("name", run_name),
            config=config,
            tags=wandb_config.get("tags", []) + [model_type],
            notes=wandb_config.get("notes", "")
        )

    # Set up tensorboard if enabled
    writer = None
    if config["logging"]["tensorboard"]:
        log_dir = Path(config["logging"]["log_dir"]) / run_name
        writer = SummaryWriter(log_dir=log_dir)

    # Create save directory for this run
    save_dir = Path(config["logging"]["save_model_dir"]) / run_name
    ensure_dir(save_dir)

    # Save merged configuration
    config_path = save_dir / "config.json"
    save_json(config, str(config_path))

    # Create datasets and dataloaders
    print("Creating datasets...")
    datasets = create_datasets(config)
    dataloaders = create_dataloaders(datasets, config)

    # Print dataset sizes
    for split, dataset in datasets.items():
        print(f"  {split} dataset: {len(dataset)} samples")

    # Create and initialize model
    print(f"Creating {model_type} model...")
    model = get_model(config, device)

    # Set up early stopping
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping"]["patience"],
        min_delta=config["training"]["early_stopping"]["min_delta"],
        mode="max"  # Higher PSNR/SSIM is better
    )

    # Training loop
    print(f"Starting training for {config['training']['num_epochs']} epochs...")
    best_metric = 0.0
    final_metrics = {}

    # Run initial validation to get baseline metrics
    if "val" in dataloaders:
        print("Running initial validation...")
        val_metrics = validate(model, dataloaders["val"], device, 0, config, writer)
        best_metric = val_metrics["psnr"]  # Use PSNR as the main metric

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")

        # Train for one epoch
        train_metrics = train_epoch(model, dataloaders["train"], device, epoch, config, writer)

        # Validate on validation set
        if "val" in dataloaders:
            val_metrics = validate(model, dataloaders["val"], device, epoch, config, writer)

            # Check for improvement
            current_metric = val_metrics["psnr"]
            improved, should_stop = early_stopping(current_metric)

            if improved:
                print(f"  New best model with PSNR: {current_metric:.4f}")
                best_metric = current_metric

                # Save best model
                model_path = save_dir / "best_model.pth"
                if hasattr(model, "save_networks"):
                    model.save_networks("best")
                else:
                    save_model(model, str(model_path))

                # Update best metrics
                final_metrics = val_metrics

            if should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Save latest model
        if epoch % config["logging"].get("save_freq", 10) == 0:
            if hasattr(model, "save_networks"):
                model.save_networks(epoch)
            else:
                model_path = save_dir / f"model_epoch_{epoch}.pth"
                save_model(model, str(model_path))

        # Update learning rate
        if hasattr(model, "update_learning_rate"):
            model.update_learning_rate(val_metrics["psnr"] if "val" in dataloaders else None)

    # Save final model
    if hasattr(model, "save_networks"):
        model.save_networks("latest")
    else:
        model_path = save_dir / "final_model.pth"
        save_model(model, str(model_path))

    # Test on test set if available
    if "test" in dataloaders:
        print("\nEvaluating on test set...")
        test_metrics = validate(model, dataloaders["test"], device, config["training"]["num_epochs"], config, writer)
        final_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    # Close wandb and tensorboard
    if config["logging"]["wandb"]["enabled"]:
        # Log final metrics
        wandb.log({"final_psnr": final_metrics.get("psnr", 0.0),
                   "final_ssim": final_metrics.get("ssim", 0.0),
                   "best_metric": best_metric})

        # Finish wandb run
        wandb.finish()

    if writer is not None:
        writer.close()

    print("\nTraining completed!")
    print(f"Best model saved with PSNR: {best_metric:.4f}")

    return final_metrics


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Load merged configuration
    config = load_model_config(args.model_type, args.base_config)

    # Override config with command line arguments
    for arg_name in ['name', 'gpu_ids', 'batch_size', 'num_epochs', 'seed', 'output_dir']:
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            if arg_name == 'batch_size':
                config['data']['batch_size'] = arg_value
            elif arg_name == 'num_epochs':
                config['training']['num_epochs'] = arg_value
            elif arg_name == 'output_dir':
                config['logging']['save_model_dir'] = os.path.join(arg_value, 'saved_models')
                config['logging']['log_dir'] = os.path.join(arg_value, 'logs')
            else:
                config[arg_name] = arg_value

    # Disable wandb if requested
    if args.no_wandb:
        config['logging']['wandb']['enabled'] = False

    # Train model
    metrics = train(config, args.model_type)

    # Print final metrics
    print("\nFinal metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()