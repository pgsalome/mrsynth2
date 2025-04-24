import os
import time
import wandb
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, Optional

from utils.config import parse_args_and_config, Config
from utils.metrics import MetricsTracker, EarlyStopping
from utils.io import ensure_dir
from data.datasets import create_datasets
from data.dataloader import create_dataloaders
from models import create_model


def setup_logging(config: Config) -> None:
    """
    Set up logging for training.

    Args:
        config: Training configuration
    """
    # Create timestamp for unique run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get run name from config or create one
    run_name = config.get("name")
    if not run_name:
        run_name = f"{config['model']['name']}_{timestamp}"

    # Update config with run name
    config["run_name"] = run_name

    # Set up directories
    log_dir = Path(config.get("logging.log_dir", "./logs")) / run_name
    save_dir = Path(config.get("logging.save_model_dir", "./saved_models")) / run_name

    # Ensure directories exist
    ensure_dir(str(log_dir))
    ensure_dir(str(save_dir))

    # Update config with directories
    config["logging.log_dir"] = str(log_dir)
    config["logging.save_model_dir"] = str(save_dir)

    # Save config for reproducibility
    config_path = log_dir / "config.json"
    config.save(config_path)

    # Set up wandb if enabled
    if config.get("logging.wandb.enabled", False):
        wandb_config = config.get("logging.wandb", {})
        wandb.init(
            project=wandb_config.get("project", "mrsynth2"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("name", run_name),
            config=config.to_dict(),
            tags=wandb_config.get("tags", []) + [config["model"]["name"]],
            notes=wandb_config.get("notes", "")
        )

    # Set up tensorboard if enabled
    if config.get("logging.tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)

    return None


def log_metrics(metrics: Dict[str, float], epoch: int, config: Config, phase: str = "train",
                writer=None) -> None:
    """
    Log metrics to console and tracking systems.

    Args:
        metrics: Dictionary of metrics
        epoch: Current epoch
        config: Training configuration
        phase: Training phase
        writer: TensorBoard writer
    """
    # Print metrics to console
    print(f"\n{phase.capitalize()} Metrics for Epoch {epoch}:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Log to wandb if enabled
    if config.get("logging.wandb.enabled", False):
        # Add epoch and phase to metrics
        wandb_metrics = {
            f"{phase}/{name}": value for name, value in metrics.items()
        }
        wandb_metrics["epoch"] = epoch
        wandb.log(wandb_metrics)

    # Log to tensorboard if enabled
    if writer is not None:
        for name, value in metrics.items():
            writer.add_scalar(f"{phase}/{name}", value, epoch)


def log_images(images: Dict[str, torch.Tensor], epoch: int, config: Config,
               writer=None) -> None:
    """
    Log images to tracking systems.

    Args:
        images: Dictionary of images
        epoch: Current epoch
        config: Training configuration
        writer: TensorBoard writer
    """
    # Only log images at specified frequency
    if epoch % config.get("logging.image_log_freq", 10) != 0:
        return

    # Log to wandb if enabled
    if config.get("logging.wandb.enabled", False):
        wandb_images = []
        for name, tensor in images.items():
            # Only log first 8 images in batch
            for i in range(min(8, tensor.size(0))):
                wandb_images.append(wandb.Image(
                    tensor[i].cpu(),
                    caption=f"{name}_{i}"
                ))

        wandb.log({f"images_epoch_{epoch}": wandb_images})

    # Log to tensorboard if enabled
    if writer is not None:
        for name, tensor in images.items():
            # Only log first 8 images in batch
            tensor = tensor[:8]
            writer.add_images(name, tensor, epoch)


def train_epoch(model, dataloader, device, epoch, config, writer=None):
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        device: Device to train on
        epoch: Current epoch
        config: Training configuration
        writer: TensorBoard writer

    Returns:
        Dictionary of training metrics
    """
    model.train()
    metrics_tracker = MetricsTracker()

    # Progress tracking
    total_batches = len(dataloader)
    start_time = time.time()
    log_interval = config.get("logging.log_interval", 100)

    for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        # Move data to device
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)

        # Forward and optimize
        model.set_input(data)
        model.optimize_parameters()

        # Get and track current losses
        losses = model.get_current_losses()
        for name, value in losses.items():
            metrics_tracker.update(name, value, data['A'].size(0))

        # Log progress
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            # Print progress and time
            progress = (batch_idx + 1) / total_batches * 100
            elapsed = time.time() - start_time
            print(f"\rEpoch {epoch} [{batch_idx + 1}/{total_batches}] ({progress:.1f}%) - "
                  f"Time: {elapsed:.2f}s", end="")

            # Calculate and print current metrics
            batch_metrics = model.get_current_losses()
            metrics_str = " | ".join([f"{n}: {v:.4f}" for n, v in batch_metrics.items()])
            print(f" | {metrics_str}", end="")

            # Log to trackers
            if config.get("logging.wandb.enabled", False):
                wandb.log({
                    f"train_batch/{n}": v for n, v in batch_metrics.items()
                })

            if writer is not None:
                step = (epoch - 1) * total_batches + batch_idx
                for n, v in batch_metrics.items():
                    writer.add_scalar(f"train_batch/{n}", v, step)

    # End of epoch line break
    print()

    # Return averaged metrics
    return metrics_tracker.compute()


def validate(model, dataloader, device, epoch, config, writer=None):
    """
    Validate model.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        device: Device to validate on
        epoch: Current epoch
        config: Validation configuration
        writer: TensorBoard writer

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker()

    # Collect some images for visualization
    vis_images = {}

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Validation")):
            # Move data to device
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)

            # Forward pass
            model.set_input(data)
            output = model.test()

            # Calculate metrics if model has validation method
            if hasattr(model, "compute_validation_metrics"):
                metrics = model.compute_validation_metrics()
                for name, value in metrics.items():
                    metrics_tracker.update(name, value, data['A'].size(0))

            # Collect first batch for visualization
            if batch_idx == 0:
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        vis_images[key] = value.cpu()

    # Log images
    log_images(vis_images, epoch, config, writer)

    # Return metrics
    return metrics_tracker.compute()


def train(config: Config) -> Dict[str, float]:
    """
    Train model according to configuration.

    Args:
        config: Training configuration

    Returns:
        Dictionary of final metrics
    """
    # Set random seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set up device
    gpu_ids = config.get("gpu_ids", "")
    if isinstance(gpu_ids, str):
        gpu_ids = [int(id) for id in gpu_ids.split(',') if id.strip()]

    if torch.cuda.is_available() and gpu_ids:
        device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Set up logging
    writer = setup_logging(config)

    # Create datasets and dataloaders
    print("Creating datasets...")
    datasets = create_datasets(config.to_dict())
    dataloaders = create_dataloaders(datasets, config.to_dict())

    # Print dataset sizes
    for split, dataset in datasets.items():
        print(f"  {split} dataset: {len(dataset)} samples")

    # Create model
    print(f"Creating {config['model']['name']} model...")
    model = create_model(config.to_dict(), device)

    # Set up early stopping
    early_stopping = None
    if config.get("training.early_stopping.enabled", False):
        patience = config.get("training.early_stopping.patience", 10)
        min_delta = config.get("training.early_stopping.min_delta", 0.0)
        mode = config.get("training.early_stopping.mode", "min")
        early_stopping = EarlyStopping(patience, min_delta, mode)

    # Training loop
    num_epochs = config.get("training.num_epochs", 100)
    best_metric = float('-inf') if early_stopping and early_stopping.mode == 'max' else float('inf')
    best_epoch = 0

    # Run initial validation if available
    validation_metric_name = config.get("training.validation_metric", "psnr")
    if "val" in dataloaders:
        print("Running initial validation...")
        val_metrics = validate(model, dataloaders["val"], device, 0, config, writer)
        log_metrics(val_metrics, 0, config, "val", writer)

        if validation_metric_name in val_metrics:
            current_metric = val_metrics[validation_metric_name]
            if early_stopping and early_stopping.mode == 'max':
                best_metric = max(best_metric, current_metric)
            else:
                best_metric = min(best_metric, current_metric)

    # Main training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train for one epoch
        train_metrics = train_epoch(model, dataloaders["train"], device, epoch, config, writer)
        log_metrics(train_metrics, epoch, config, "train", writer)

        # Validate
        if "val" in dataloaders:
            val_metrics = validate(model, dataloaders["val"], device, epoch, config, writer)
            log_metrics(val_metrics, epoch, config, "val", writer)

            # Check for improvement
            if validation_metric_name in val_metrics:
                current_metric = val_metrics[validation_metric_name]
                improved = False

                if early_stopping:
                    improved, should_stop = early_stopping(current_metric)

                    if should_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
                else:
                    # Manual improvement check
                    if early_stopping and early_stopping.mode == 'max':
                        improved = current_metric > best_metric
                    else:
                        improved = current_metric < best_metric

                if improved:
                    print(f"  New best model with {validation_metric_name}: {current_metric:.4f}")
                    best_metric = current_metric
                    best_epoch = epoch

                    # Save best model
                    save_path = Path(config.get("logging.save_model_dir")) / "best_model.pth"
                    if hasattr(model, "save_networks"):
                        model.save_networks("best")
                    else:
                        torch.save(model.state_dict(), save_path)

        # Save checkpoint
        if epoch % config.get("logging.save_freq", 10) == 0:
            # Save model checkpoint
            if hasattr(model, "save_networks"):
                model.save_networks(f"epoch_{epoch}")
            else:
                save_path = Path(config.get("logging.save_model_dir")) / f"model_epoch_{epoch}.pth"
                torch.save(model.state_dict(), save_path)

        # Update learning rate if scheduler exists
        if hasattr(model, "update_learning_rate"):
            if "val" in dataloaders and validation_metric_name in val_metrics:
                model.update_learning_rate(val_metrics[validation_metric_name])
            else:
                model.update_learning_rate()

    # Final evaluation on test set
    final_metrics = {}
    if "test" in dataloaders:
        print("\nEvaluating on test set...")
        test_metrics = validate(model, dataloaders["test"], device, num_epochs, config, writer)
        log_metrics(test_metrics, num_epochs, config, "test", writer)
        final_metrics.update(test_metrics)

    # Save final model
    save_path = Path(config.get("logging.save_model_dir")) / "final_model.pth"
    if hasattr(model, "save_networks"):
        model.save_networks("final")
    else:
        torch.save(model.state_dict(), save_path)

    # Clean up
    if writer is not None:
        writer.close()

    if config.get("logging.wandb.enabled", False):
        wandb.finish()

    # Print final statistics
    print(f"\nTraining completed!")
    print(f"Best {validation_metric_name}: {best_metric:.4f} at epoch {best_epoch}")

    return final_metrics


def main():
    """Main function for training."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train image-to-image translation models")

    # Add model-specific arguments
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["cyclegan", "pix2pix", "diffusion", "latent_diffusion", "vae"],
                        help="Type of model to train")
    parser.add_argument("--base_config", type=str, default=None,
                        help="Path to base config file")
    parser.add_argument("--dataset_dir", type=str,
                        help="Directory containing dataset")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int,
                        help="Number of training epochs")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save outputs")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")

    # Parse arguments and load config
    config = parse_args_and_config(parser)

    # Override config settings from command line
    if args.no_wandb:
        config["logging.wandb.enabled"] = False

    if args.output_dir:
        config["logging.save_model_dir"] = os.path.join(args.output_dir, "saved_models")
        config["logging.log_dir"] = os.path.join(args.output_dir, "logs")

    # Train model
    train(config)


if __name__ == "__main__":
    main()