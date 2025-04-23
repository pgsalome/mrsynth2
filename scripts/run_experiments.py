import os
import json
import copy
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

# Import Optuna for Bayesian optimization
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.trial import Trial

# Import wandb for handling runs
import wandb

from src.utils.io import ensure_dir
from src.utils.config import load_model_config
from scripts.train import train


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def define_parameter_space_cyclegan(trial: Trial) -> Dict[str, Any]:
    """Define hyperparameter search space for CycleGAN."""
    params = {}

    # Training parameters
    params["data.batch_size"] = trial.suggest_categorical("data.batch_size", [1, 2, 4, 8])
    params["training.optimizer.lr"] = trial.suggest_float("training.optimizer.lr", 1e-5, 1e-3, log=True)
    params["training.optimizer.beta1"] = trial.suggest_float("training.optimizer.beta1", 0.1, 0.9)
    params["training.loss.lambda_A"] = trial.suggest_float("training.loss.lambda_A", 5.0, 15.0)
    params["training.loss.lambda_B"] = trial.suggest_float("training.loss.lambda_B", 5.0, 15.0)
    params["training.loss.lambda_identity"] = trial.suggest_float("training.loss.lambda_identity", 0.0, 1.0)

    # Model architecture
    params["model.G_A.name"] = trial.suggest_categorical(
        "model.G_A.name", ["resnet_9blocks", "resnet_6blocks", "attention_resnet"]
    )
    params["model.G_B.name"] = params["model.G_A.name"]  # Use same generator architecture for both directions

    params["model.D_A.name"] = trial.suggest_categorical(
        "model.D_A.name", ["basic", "n_layers", "spectral"]
    )
    params["model.D_B.name"] = params["model.D_A.name"]  # Use same discriminator architecture

    # GAN loss type
    params["training.loss.gan_mode"] = trial.suggest_categorical(
        "training.loss.gan_mode", ["vanilla", "lsgan"]
    )

    # Optional perceptual loss
    use_perceptual = trial.suggest_categorical("use_perceptual_loss", [True, False])
    if use_perceptual:
        params["training.loss.lambda_perceptual"] = trial.suggest_float(
            "training.loss.lambda_perceptual", 0.1, 10.0, log=True
        )
    else:
        params["training.loss.lambda_perceptual"] = 0.0

    # Learning rate scheduler
    scheduler_type = trial.suggest_categorical("scheduler_type", ["step", "cosine", "plateau"])
    params["training.scheduler.name"] = scheduler_type

    if scheduler_type == "step":
        params["training.scheduler.params.step_size"] = trial.suggest_int("training.scheduler.params.step_size", 10, 50)
        params["training.scheduler.params.gamma"] = trial.suggest_float("training.scheduler.params.gamma", 0.1, 0.5)
    elif scheduler_type == "plateau":
        params["training.scheduler.params.factor"] = trial.suggest_float("training.scheduler.params.factor", 0.1, 0.5)
        params["training.scheduler.params.patience"] = trial.suggest_int("training.scheduler.params.patience", 5, 15)

    return params


def define_parameter_space_pix2pix(trial: Trial) -> Dict[str, Any]:
    """Define hyperparameter search space for Pix2Pix."""
    params = {}

    # Training parameters
    params["data.batch_size"] = trial.suggest_categorical("data.batch_size", [1, 2, 4, 8, 16])
    params["training.optimizer.lr"] = trial.suggest_float("training.optimizer.lr", 1e-5, 1e-3, log=True)
    params["training.optimizer.beta1"] = trial.suggest_float("training.optimizer.beta1", 0.1, 0.9)

    # Model architecture
    params["model.G_A.name"] = trial.suggest_categorical(
        "model.G_A.name", ["unet_256", "unet_128", "resnet_9blocks"]
    )

    params["model.G_A.use_dropout"] = trial.suggest_categorical("model.G_A.use_dropout", [True, False])

    params["model.D_A.name"] = trial.suggest_categorical(
        "model.D_A.name", ["basic", "n_layers"]
    )

    # Loss weights
    params["training.loss.gan_mode"] = trial.suggest_categorical(
        "training.loss.gan_mode", ["vanilla", "lsgan"]
    )
    params["training.loss.lambda_L1"] = trial.suggest_float("training.loss.lambda_L1", 50.0, 150.0)

    # Optional perceptual loss
    use_perceptual = trial.suggest_categorical("use_perceptual_loss", [True, False])
    if use_perceptual:
        params["training.loss.lambda_perceptual"] = trial.suggest_float(
            "training.loss.lambda_perceptual", 0.1, 20.0, log=True
        )
    else:
        params["training.loss.lambda_perceptual"] = 0.0

    return params


def define_parameter_space_diffusion(trial: Trial) -> Dict[str, Any]:
    """Define hyperparameter search space for Diffusion model."""
    params = {}

    # Training parameters
    params["data.batch_size"] = trial.suggest_categorical("data.batch_size", [1, 2, 4, 8])
    params["training.optimizer.lr"] = trial.suggest_float("training.optimizer.lr", 1e-5, 1e-3, log=True)

    # Diffusion parameters
    params["model.diffusion.json.beta_schedule"] = trial.suggest_categorical(
        "model.diffusion.json.beta_schedule", ["linear", "cosine", "quadratic"]
    )
    params["model.diffusion.json.timesteps"] = trial.suggest_categorical(
        "model.diffusion.json.timesteps", [500, 1000, 2000]
    )
    params["model.diffusion.json.use_snr_weighting"] = trial.suggest_categorical(
        "model.diffusion.json.use_snr_weighting", [True, False]
    )

    # UNet architecture
    params["model.unet.ngf"] = trial.suggest_categorical("model.unet.ngf", [32, 64, 128])
    params["model.unet.use_dropout"] = trial.suggest_categorical("model.unet.use_dropout", [True, False])

    return params


def define_parameter_space_latent_diffusion(trial: Trial) -> Dict[str, Any]:
    """Define hyperparameter search space for Latent Diffusion model."""
    params = {}

    # Training parameters
    params["data.batch_size"] = trial.suggest_categorical("data.batch_size", [1, 2, 4, 8])
    params["training.optimizer.lr"] = trial.suggest_float("training.optimizer.lr", 1e-5, 1e-3, log=True)

    # Latent dimension
    params["model.latent_dim"] = trial.suggest_categorical("model.latent_dim", [3, 4, 8])

    # Diffusion parameters
    params["model.diffusion.json.beta_schedule"] = trial.suggest_categorical(
        "model.diffusion.json.beta_schedule", ["linear", "cosine", "quadratic"]
    )
    params["model.diffusion.json.timesteps"] = trial.suggest_categorical(
        "model.diffusion.json.timesteps", [500, 1000, 2000]
    )
    params["model.diffusion.json.use_snr_weighting"] = trial.suggest_categorical(
        "model.diffusion.json.use_snr_weighting", [True, False]
    )
    params["model.diffusion.json.latent_conditioning"] = trial.suggest_categorical(
        "model.diffusion.json.latent_conditioning", [True, False]
    )

    # UNet architecture
    params["model.unet.ngf"] = trial.suggest_categorical("model.unet.ngf", [32, 64, 128])
    params["model.unet.use_dropout"] = trial.suggest_categorical("model.unet.use_dropout", [True, False])

    return params


def define_parameter_space_vae(trial: Trial) -> Dict[str, Any]:
    """Define hyperparameter search space for VAE model."""
    params = {}

    # Training parameters
    params["data.batch_size"] = trial.suggest_categorical("data.batch_size", [4, 8, 16, 32])
    params["training.optimizer.lr"] = trial.suggest_float("training.optimizer.lr", 1e-5, 1e-3, log=True)

    # VAE parameters
    params["model.vae.latent_dim"] = trial.suggest_categorical("model.vae.latent_dim", [8, 16, 32, 64])
    params["model.vae.kl_weight"] = trial.suggest_float("model.vae.kl_weight", 1e-5, 1e-2, log=True)
    params["model.vae.recon_loss"] = trial.suggest_categorical("model.vae.recon_loss", ["l1", "l2"])

    # Encoder/decoder architecture
    params["model.vae.encoder.ngf"] = trial.suggest_categorical("model.vae.encoder.ngf", [32, 64, 128])
    params["model.vae.decoder.ngf"] = params["model.vae.encoder.ngf"]  # Keep same ngf for encoder/decoder

    return params


def define_parameter_space(trial: Trial, model_type: str) -> Dict[str, Any]:
    """
    Define the hyperparameters search space for an Optuna trial.

    Args:
        trial: Optuna trial object
        model_type: Model type to optimize

    Returns:
        Dictionary of sampled parameters
    """
    if model_type == "cyclegan":
        return define_parameter_space_cyclegan(trial)
    elif model_type == "pix2pix.json":
        return define_parameter_space_pix2pix(trial)
    elif model_type == "diffusion.json":
        return define_parameter_space_diffusion(trial)
    elif model_type == "latent_diffusion":
        return define_parameter_space_latent_diffusion(trial)
    elif model_type == "vae":
        return define_parameter_space_vae(trial)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def update_config_with_params(base_config: Dict[str, Any], param_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with parameters.

    Args:
        base_config: Base configuration dictionary
        param_dict: Dictionary of parameters (with nested paths as keys)

    Returns:
        Updated configuration dictionary
    """
    config = copy.deepcopy(base_config)

    for param_name, param_value in param_dict.items():
        # Convert period-separated path to nested dictionary keys
        keys = param_name.split(".")

        # Navigate to the right level in the config
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = param_value

    return config


def objective(trial, base_config, model_type):
    """Optuna objective function"""
    # Sample hyperparameters for the specific model type
    params = define_parameter_space(trial, model_type)

    # Update base config with sampled parameters
    config = update_config_with_params(base_config, params)

    # Set the config number for this trial
    config['config_num'] = trial.number

    # Create unique run name for wandb
    run_name = f"{model_type}_optuna_trial_{trial.number}"
    if config["logging"]["wandb"]["name"] is None:
        config["logging"]["wandb"]["name"] = run_name

    # Add trial details to wandb config
    config["trial_number"] = trial.number

    # Update wandb config
    if "logging" in config and "wandb" in config["logging"] and config["logging"]["wandb"]["enabled"]:
        # Terminate any existing wandb run
        if wandb.run is not None:
            wandb.finish()

        # Add trial-specific tags
        if "tags" not in config["logging"]["wandb"]:
            config["logging"]["wandb"]["tags"] = []
        config["logging"]["wandb"]["tags"].extend(["optuna", f"trial_{trial.number}", model_type])

    # Print the configuration
    print(f"\nOptuna Trial {trial.number}")
    print(f"Parameters: {json.dumps(params, indent=2)}")

    # Save the configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dir = Path(config.get("output_dir", "./optuna_configs"))
    ensure_dir(config_dir)
    config_path = config_dir / f"{model_type}_config_trial_{trial.number}_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)

    try:
        # Run training
        metrics = train(config, model_type)

        # Get the optimization metric (default to PSNR)
        if "val_psnr" in metrics:
            score = metrics["val_psnr"]
        elif "psnr" in metrics:
            score = metrics["psnr"]
        else:
            # Fallback to first available metric
            score = list(metrics.values())[0]

        # Record the results
        print(f"Trial {trial.number} completed with score: {score:.4f}")

        # Ensure wandb run is finished
        if wandb.run is not None:
            wandb.finish()

        return score

    except Exception as e:
        print(f"Error in trial {trial.number}: {str(e)}")

        # Ensure wandb run is finished even if there's an error
        if wandb.run is not None:
            wandb.finish()

        # Return a small value for failed trials
        return 0.01


def run_bayesian_optimization(
        model_type: str,
        base_config_path: str,
        output_dir: str,
        n_trials: int = 20,
        random_state: int = 42,
        results_file: str = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        n_jobs: int = 1
):
    """
    Run Bayesian optimization for hyperparameter tuning using Optuna.

    Args:
        model_type: Type of model to optimize
        base_config_path: Path to base configuration file
        output_dir: Directory to save experiment configurations
        n_trials: Number of optimization trials
        random_state: Random seed
        results_file: Path to save experiment results
        study_name: Name for the Optuna study (for persistence)
        storage: Storage URL for Optuna (for persistence)
        n_jobs: Number of parallel jobs
    """
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Create results file path if not provided
    if results_file is None:
        results_file = f"{model_type}_optuna_results.csv"
    results_file_path = Path(output_dir) / results_file

    # Load the base configuration
    base_config = load_model_config(model_type, base_config_path)

    # Update output directory in config
    base_config["logging"]["save_model_dir"] = os.path.join(output_dir, "saved_models")
    base_config["logging"]["log_dir"] = os.path.join(output_dir, "logs")
    base_config["output_dir"] = output_dir

    # Timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create or load an Optuna study
    if study_name and storage:
        full_study_name = f"{model_type}_{study_name}"
        # Resume or create a persistent study
        study = optuna.create_study(
            study_name=full_study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
    else:
        # Create a new study for this run only
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )

    print(f"Starting Optuna optimization for model type {model_type} with {n_trials} trials...")

    try:
        # Run optimization
        study.optimize(
            lambda trial: objective(trial, base_config, model_type),
            n_trials=n_trials,
            n_jobs=n_jobs
        )

        # Generate and save visualizations
        try:
            # Plot optimization history
            history_fig = plot_optimization_history(study)
            history_fig_path = Path(output_dir) / f"{model_type}_optimization_history_{timestamp}.png"
            history_fig.write_image(str(history_fig_path))

            # Plot parameter importance
            importance_fig = plot_param_importances(study)
            importance_fig_path = Path(output_dir) / f"{model_type}_param_importances_{timestamp}.png"
            importance_fig.write_image(str(importance_fig_path))

        except Exception as e:
            print(f"Warning: Could not generate Optuna visualization plots: {e}")

        # Get best parameters from the study
        best_trial = study.best_trial

        # Print and save the best results
        print("\nOptuna Optimization Results:")
        print(f"Best score: {best_trial.value:.4f}")
        print("Best parameters:")
        for param_name, param_value in best_trial.params.items():
            print(f"  {param_name}: {param_value}")

        # Create a comprehensive report
        report = {
            "model_type": model_type,
            "timestamp": timestamp,
            "best_score": float(best_trial.value),
            "best_parameters": best_trial.params,
            "total_trials": len(study.trials),
            "runtime_info": {
                "start_time": timestamp,
                "end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "n_trials": n_trials,
                "random_state": random_state
            }
        }

        # Save the report
        report_path = Path(output_dir) / f"{model_type}_optimization_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)

        # Create config with the best parameters
        best_config = update_config_with_params(base_config, best_trial.params)
        best_config_path = Path(output_dir) / f"{model_type}_best_config_{timestamp}.json"
        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=2, cls=NumpyEncoder)

        # Extract all trial results to a DataFrame
        results_df = pd.DataFrame({
            "trial_number": [t.number for t in study.trials],
            "value": [t.value if t.value is not None else float('nan') for t in study.trials],
            "state": [t.state.name for t in study.trials],
            "datetime_start": [t.datetime_start.strftime("%Y-%m-%d %H:%M:%S") if t.datetime_start else None for t in
                               study.trials],
            "datetime_complete": [t.datetime_complete.strftime("%Y-%m-%d %H:%M:%S") if t.datetime_complete else None for
                                  t in study.trials]
        })

        # Add parameters to the DataFrame
        for t in study.trials:
            for param_name, param_value in t.params.items():
                if param_name not in results_df.columns:
                    results_df[param_name] = None
                results_df.loc[results_df.trial_number == t.number, param_name] = param_value

        # Save to CSV
        results_df.to_csv(results_file_path, index=False)
        print(f"Trial results saved to {results_file_path}")

        return best_config, best_trial.value, results_df

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print("Saving current best results...")

        # Save what we have so far
        interrupted_path = Path(output_dir) / f"{model_type}_interrupted_best_config_{timestamp}.json"
        if len(study.trials) > 0:
            best_trial_so_far = study.best_trial
            best_config_so_far = update_config_with_params(base_config, best_trial_so_far.params)
            with open(interrupted_path, "w") as f:
                json.dump(best_config_so_far, f, indent=2, cls=NumpyEncoder)
            print(f"Interrupted best config saved to {interrupted_path}")
        else:
            with open(interrupted_path, "w") as f:
                json.dump({"status": "interrupted_before_best_found"}, f, indent=2, cls=NumpyEncoder)
            print("No completed trials found.")

        # Extract the results we have so far
        if len(study.trials) > 0:
            results_df = pd.DataFrame({
                "trial_number": [t.number for t in study.trials],
                "value": [t.value if t.value is not None else float('nan') for t in study.trials],
                "state": [t.state.name for t in study.trials],
                "datetime_start": [t.datetime_start.strftime("%Y-%m-%d %H:%M:%S") if t.datetime_start else None for t in
                                   study.trials],
                "datetime_complete": [t.datetime_complete.strftime("%Y-%m-%d %H:%M:%S") if t.datetime_complete else None
                                      for t in study.trials]
            })

            # Add parameters to the DataFrame
            for t in study.trials:
                if t.params:
                    for param_name, param_value in t.params.items():
                        if param_name not in results_df.columns:
                            results_df[param_name] = None
                        results_df.loc[results_df.trial_number == t.number, param_name] = param_value

            # Save to CSV
            results_df.to_csv(results_file_path, index=False)
            print(f"Partial trial results saved to {results_file_path}")

            return None, None, results_df

        return None, None, None

    finally:
        # Ensure all wandb runs are closed
        if wandb.run is not None:
            wandb.finish()


def run_all_model_optimizations(
        model_types: List[str],
        base_config_path: str,
        output_dir: str,
        n_trials_per_model: int = 20,
        random_state: int = 42,
        storage: Optional[str] = None,
        n_jobs: int = 1
):
    """
    Run Bayesian optimization for multiple model types.

    Args:
        model_types: List of model types to optimize
        base_config_path: Path to base configuration file
        output_dir: Directory to save experiment configurations
        n_trials_per_model: Number of optimization trials per model
        random_state: Random seed
        storage: Storage URL for Optuna (for persistence)
        n_jobs: Number of parallel jobs
    """
    # Store results for all models
    all_results = {}

    # Create a shared study name based on timestamp
    shared_study_name = f"multimodel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run optimization for each model type
    for model_type in model_types:
        print(f"\n{'=' * 80}")
        print(f"Starting optimization for {model_type}")
        print(f"{'=' * 80}")

        # Create model-specific output directory
        model_output_dir = os.path.join(output_dir, model_type)
        ensure_dir(model_output_dir)

        # Run optimization
        best_config, best_score, results_df = run_bayesian_optimization(
            model_type=model_type,
            base_config_path=base_config_path,
            output_dir=model_output_dir,
            n_trials=n_trials_per_model,
            random_state=random_state,
            study_name=shared_study_name if storage else None,
            storage=storage,
            n_jobs=n_jobs
        )

        # Store results
        all_results[model_type] = {
            'best_score': best_score,
            'best_config': best_config,
            'results_df': results_df
        }

    # Compare models
    print("\nModel Comparison Results:")
    print(f"{'Model Type':<20} {'Best Score':<15}")
    print(f"{'-' * 35}")

    best_model = None
    best_score = float('-inf')

    for model_type, result in all_results.items():
        if result['best_score'] is not None:
            score = result['best_score']
            print(f"{model_type:<20} {score:<15.4f}")

            if score > best_score:
                best_score = score
                best_model = model_type

    if best_model:
        print(f"\nBest model: {best_model} with score {best_score:.4f}")

        # Save best overall model config
        best_overall_config = all_results[best_model]['best_config']
        best_overall_path = os.path.join(output_dir, f"best_overall_{best_model}_config.json")
        with open(best_overall_path, 'w') as f:
            json.dump(best_overall_config, f, indent=2, cls=NumpyEncoder)

        print(f"Best overall configuration saved to {best_overall_path}")
    else:
        print("\nNo valid results found for any model type.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Bayesian optimization for mrsynth2 models")
    parser.add_argument("--base_config", type=str, default="config/base.json",
                        help="Path to base config file")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Directory to save experiment outputs")
    parser.add_argument("--mode", type=str, choices=["single", "all"], default="single",
                        help="Mode: 'single' for optimizing one model, 'all' for all models")
    parser.add_argument("--model_type", type=str,
                        choices=["cyclegan", "pix2pix.json", "diffusion.json", "latent_diffusion", "vae"],
                        default="cyclegan", help="Model type to optimize (for 'single' mode)")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials per model")
    parser.add_argument("--storage", type=str, help="Storage URL for Optuna (e.g., 'sqlite:///optuna.db')")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    ensure_dir(args.output_dir)

    if args.mode == "single":
        # Run optimization for a single model type
        run_bayesian_optimization(
            model_type=args.model_type,
            base_config_path=args.base_config,
            output_dir=args.output_dir,
            n_trials=args.n_trials,
            random_state=args.seed,
            storage=args.storage,
            n_jobs=args.n_jobs
        )
    elif args.mode == "all":
        # Run optimization for all model types
        model_types = ["cyclegan", "pix2pix.json", "diffusion.json", "latent_diffusion", "vae"]
        run_all_model_optimizations(
            model_types=model_types,
            base_config_path=args.base_config,
            output_dir=args.output_dir,
            n_trials_per_model=args.n_trials,
            random_state=args.seed,
            storage=args.storage,
            n_jobs=args.n_jobs
        )


if __name__ == "__main__":
    main()