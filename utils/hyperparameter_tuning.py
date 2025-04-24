import optuna
import wandb
from training.train import train_model
import os
import json


def define_parameter_space():
    """Load the hyperparameter space from config file"""
    with open('config/hyperparameter_space.json', 'r') as f:
        return json.load(f)


def objective(trial, base_config):
    """Optuna objective function"""
    # Get hyperparameter space
    param_space = define_parameter_space()

    # Sample hyperparameters
    hparams = {}
    for param, settings in param_space.items():
        if settings['type'] == 'categorical':
            hparams[param] = trial.suggest_categorical(param, settings['values'])
        elif settings['type'] == 'float':
            hparams[param] = trial.suggest_float(param, settings['min'], settings['max'],
                                                 log=settings.get('log', False))
        elif settings['type'] == 'int':
            hparams[param] = trial.suggest_int(param, settings['min'], settings['max'], log=settings.get('log', False))

    # Update base config with sampled hyperparameters
    config = base_config.copy()
    for key, value in hparams.items():
        config[key] = value

    # Create unique run name for wandb
    run_name = f"trial_{trial.number}"

    # Initialize wandb
    wandb_run = wandb.init(project=config['wandb_project_name'], name=run_name, config=config)

    # Train model with these hyperparameters
    metrics = train_model(config)

    # Close wandb run
    wandb_run.finish()

    # Return the metric to optimize
    return metrics['validation_metric']


def tune_hyperparameters(base_config, n_trials=50, study_name="cyclegan_study"):
    """Run Optuna hyperparameter tuning"""
    # Create storage for Optuna study
    storage_name = f"sqlite:///{study_name}.db"

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize"
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config),
        n_trials=n_trials
    )

    # Print results
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save best hyperparameters
    best_params = best_trial.params
    with open('config/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    return best_params