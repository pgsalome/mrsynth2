import time
import os
import torch
import sys
import json
import argparse
from tqdm import tqdm

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import create_dataset
from models import create_model
from utils.visualizer import Visualizer
from utils.cache import check_cache, get_cached_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CycleGAN/pix2pix')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataroot', type=str, help='Path to dataset (overrides config)')
    parser.add_argument('--name', type=str, help='Name of the experiment (overrides config)')
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs to use (overrides config)')
    parser.add_argument('--checkpoints_dir', type=str, help='Where to save checkpoints (overrides config)')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging (overrides config)')
    parser.add_argument('--use_cache', action='store_true', help='Use cached preprocessed data')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def train_model(opt):
    """Train model with given options"""
    # Create dataset
    print("Creating dataset...")
    # Check cache if enabled
    dataset_cache_key = f"dataset_{opt['dataroot']}_{opt['dataset_mode']}_{opt['phase']}"
    if opt.get('use_cache', False) and check_cache(dataset_cache_key):
        print(f"Using cached dataset: {dataset_cache_key}")
        dataset = get_cached_data(dataset_cache_key)
    else:
        dataset = create_dataset(opt)

    dataset_size = len(dataset)
    print(f'The number of training images = {dataset_size}')

    # Create validation dataset if enabled
    if opt.get('enable_validation', False):
        opt_val = opt.copy()
        opt_val['phase'] = 'val'
        val_dataset = create_dataset(opt_val)
        val_dataset_size = len(val_dataset)
        print(f'The number of validation images = {val_dataset_size}')
    else:
        val_dataset = None

    # Create model
    print("Creating model...")
    model = create_model(opt)
    model.setup(opt)

    # Create visualizer
    visualizer = Visualizer(opt)

    # Initialize wandb if enabled
    if opt.get('use_wandb', False):
        import wandb
        wandb.init(project=opt.get('wandb_project_name', 'pytorch-CycleGAN-and-pix2pix'),
                   name=opt.get('name', 'experiment'),
                   config=opt)

    # Initialize metrics
    metrics = {
        'validation_metric': 0.0,
        'best_epoch': 0
    }

    # Start training
    print("Starting training...")
    total_iters = 0

    # Track best model performance
    best_metric = float('-inf')
    best_epoch = 0

    # Training loop
    for epoch in range(opt.get('epoch_count', 1), opt.get('n_epochs', 100) + opt.get('n_epochs_decay', 100) + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        # Train for one epoch
        model.train()
        for i, data in enumerate(tqdm(dataset, desc=f"Epoch {epoch}")):
            iter_start_time = time.time()

            if total_iters % opt.get('print_freq', 100) == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.get('batch_size', 1)
            epoch_iter += opt.get('batch_size', 1)

            # Process and optimize
            model.set_input(data)
            model.optimize_parameters()

            # Display and logging
            if total_iters % opt.get('display_freq', 400) == 0:
                save_result = total_iters % opt.get('update_html_freq', 1000) == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # Print losses
            if total_iters % opt.get('print_freq', 100) == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.get('batch_size', 1)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.get('display_id', 0) > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                # Log to wandb
                if opt.get('use_wandb', False):
                    wandb.log(losses)

            # Save latest model
            if total_iters % opt.get('save_latest_freq', 5000) == 0:
                print(f'saving the latest model (epoch {epoch}, total_iters {total_iters})')
                save_suffix = 'iter_%d' % total_iters if opt.get('save_by_iter', False) else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # Validation
        if val_dataset and epoch % opt.get('val_freq', 1) == 0:
            model.eval()
            print(f"Performing validation at the end of epoch {epoch}")
            val_losses = model.validate(val_dataset)

            # Log validation metrics
            print(f"Validation results: {val_losses}")
            if opt.get('use_wandb', False):
                wandb.log(val_losses)

            # Update learning rate
            model.update_learning_rate(val_losses.get('PSNR', None))

            # Track best model
            current_metric = val_losses.get('PSNR', 0)
            metrics['validation_metric'] = current_metric

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                metrics['best_epoch'] = best_epoch
                print(f"New best model with PSNR: {best_metric:.4f}")
                model.save_networks('best')

        # Save model for each epoch
        if epoch % opt.get('save_epoch_freq', 5) == 0:
            print(f'saving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)

        print(
            f'End of epoch {epoch} / {opt.get("n_epochs", 100) + opt.get("n_epochs_decay", 100)} \t Time Taken: {time.time() - epoch_start_time:.0f} sec')

    return metrics


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    for arg in vars(args):
        if getattr(args, arg) is not None and arg != 'config':
            config[arg] = getattr(args, arg)

    # Train model
    metrics = train_model(config)

    print(
        f"Training completed. Best validation metric: {metrics['validation_metric']:.4f} at epoch {metrics['best_epoch']}")


if __name__ == '__main__':
    main()