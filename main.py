import argparse
import os
import sys
import json
import subprocess


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CycleGAN and pix2pix main script')

    # Main options
    parser.add_argument('operation', type=str, choices=['preprocess', 'train', 'test', 'tune'],
                        help='Operation to perform: preprocess, train, test, or tune')
    parser.add_argument('--config', type=str, default='config/train_config.json',
                        help='Path to config file')

    # Common options
    parser.add_argument('--dataroot', type=str, help='Path to dataset (overrides config)')
    parser.add_argument('--name', type=str, help='Name of the experiment (overrides config)')
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs to use (overrides config)')

    # Preprocessing options
    parser.add_argument('--output_dir', type=str, help='Output directory for preprocessed data')
    parser.add_argument('--preprocess_mode', type=str,
                        help='Preprocessing mode [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--force_preprocess', action='store_true',
                        help='Force preprocessing even if cache exists')

    # Training options
    parser.add_argument('--checkpoints_dir', type=str, help='Where to save checkpoints')
    parser.add_argument('--epoch_count', type=int, help='Starting epoch count')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs with initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, help='Number of epochs to decay learning rate to zero')

    # Test options
    parser.add_argument('--results_dir', type=str, help='Where to save results')
    parser.add_argument('--aspect_ratio', type=float, help='Aspect ratio of result images')
    parser.add_argument('--phase', type=str, help='Train, val, test, etc')
    parser.add_argument('--eval', action='store_true', help='Use eval mode during test')
    parser.add_argument('--num_test', type=int, help='How many test images to run')

    # Tuning options
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='cyclegan_study', help='Optuna study name')

    # Logging options
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb_project_name', type=str, help='Project name for wandb')

    # Cache options
    parser.add_argument('--use_cache', action='store_true', help='Use cached preprocessed data')
    parser.add_argument('--clear_cache', action='store_true', help='Clear cache before running')

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def update_config(config, args):
    """Update config with command line arguments"""
    for arg in vars(args):
        if getattr(args, arg) is not None and arg not in ['operation', 'config']:
            config[arg] = getattr(args, arg)
    return config


def preprocess(config):
    """Run preprocessing"""
    from preprocessing.preprocess import main as preprocess_main
    print("Running preprocessing...")

    # Prepare arguments
    sys.argv = [
        'preprocess.py',
        '--config', config['config'],
    ]

    # Add additional arguments
    if 'dataroot' in config:
        sys.argv.extend(['--input_dir', config['dataroot']])
    if 'output_dir' in config:
        sys.argv.extend(['--output_dir', config['output_dir']])
    if 'phase' in config:
        sys.argv.extend(['--phase', config['phase']])
    if 'preprocess_mode' in config:
        sys.argv.extend(['--preprocess', config['preprocess_mode']])
    if 'force_preprocess' in config and config['force_preprocess']:
        sys.argv.append('--force')

    # Run preprocessing
    preprocess_main()


def train(config):
    """Run training"""
    from training.train import main as train_main
    print("Running training...")

    # Prepare arguments
    sys.argv = [
        'train.py',
        '--config', config['config'],
    ]

    # Run training
    train_main()


def test(config):
    """Run testing"""
    from training.test import main as test_main
    print("Running testing...")

    # Prepare arguments
    sys.argv = [
        'test.py',
        '--config', config['config'],
    ]

    # Run testing
    test_main()


def tune(config):
    """Run hyperparameter tuning"""
    from utils.hyperparameter_tuning import tune_hyperparameters
    print("Running hyperparameter tuning...")

    # Run tuning
    best_params = tune_hyperparameters(
        config,
        n_trials=config.get('n_trials', 50),
        study_name=config.get('study_name', 'cyclegan_study')
    )

    print(f"Tuning completed. Best parameters: {best_params}")


def handle_cache(args):
    """Handle cache operations"""
    if args.clear_cache:
        from utils.cache import clear_cache
        print("Clearing cache...")
        clear_cache()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Update config with command line arguments
    config = update_config(config, args)

    # Handle cache operations
    handle_cache(args)

    # Run selected operation
    if args.operation == 'preprocess':
        preprocess(config)
    elif args.operation == 'train':
        train(config)
    elif args.operation == 'test':
        test(config)
    elif args.operation == 'tune':
        tune(config)
    else:
        print(f"Unknown operation: {args.operation}")


if __name__ == '__main__':
    main()