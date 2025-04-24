from typing import Dict, Any
import optuna


def define_hyperparameter_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define hyperparameter search space for CycleGAN.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary with sampled hyperparameters
    """
    params = {}

    # Generator architecture
    params["model.G_A.name"] = trial.suggest_categorical(
        "generator_architecture",
        ["resnet_9blocks", "resnet_6blocks", "attention_resnet"]
    )
    params["model.G_B.name"] = params["model.G_A.name"]  # Use same for both generators

    params["model.G_A.ngf"] = trial.suggest_categorical(
        "generator_filters",
        [32, 64, 128]
    )
    params["model.G_B.ngf"] = params["model.G_A.ngf"]  # Use same for both generators

    params["model.G_A.use_dropout"] = trial.suggest_categorical(
        "generator_dropout",
        [True, False]
    )
    params["model.G_B.use_dropout"] = params["model.G_A.use_dropout"]

    # Discriminator architecture
    params["model.D_A.name"] = trial.suggest_categorical(
        "discriminator_architecture",
        ["basic", "n_layers"]
    )
    params["model.D_B.name"] = params["model.D_A.name"]  # Use same for both discriminators

    params["model.D_A.ndf"] = trial.suggest_categorical(
        "discriminator_filters",
        [32, 64, 128]
    )
    params["model.D_B.ndf"] = params["model.D_A.ndf"]  # Use same for both discriminators

    params["model.D_A.n_layers"] = trial.suggest_int(
        "discriminator_layers",
        2, 4
    )
    params["model.D_B.n_layers"] = params["model.D_A.n_layers"]

    # Training parameters
    params["data.batch_size"] = trial.suggest_categorical(
        "batch_size",
        [1, 2, 4, 8]
    )

    params["training.optimizer.lr"] = trial.suggest_float(
        "learning_rate",
        1e-5, 1e-3, log=True
    )

    params["training.optimizer.beta1"] = trial.suggest_float(
        "beta1",
        0.1, 0.9
    )

    # Loss weights
    params["training.loss.gan_mode"] = trial.suggest_categorical(
        "gan_mode",
        ["vanilla", "lsgan"]
    )

    params["training.loss.lambda_A"] = trial.suggest_float(
        "lambda_A",
        5.0, 15.0
    )

    params["training.loss.lambda_B"] = trial.suggest_float(
        "lambda_B",
        5.0, 15.0
    )

    params["training.loss.lambda_identity"] = trial.suggest_float(
        "lambda_identity",
        0.0, 1.0
    )

    # Optional perceptual loss
    use_perceptual = trial.suggest_categorical(
        "use_perceptual_loss",
        [True, False]
    )

    if use_perceptual:
        params["training.loss.lambda_perceptual"] = trial.suggest_float(
            "lambda_perceptual",
            0.1, 10.0, log=True
        )
    else:
        params["training.loss.lambda_perceptual"] = 0.0

    # Learning rate scheduler
    params["training.scheduler.name"] = trial.suggest_categorical(
        "scheduler",
        ["step", "cosine", "plateau"]
    )

    if params["training.scheduler.name"] == "step":
        params["training.scheduler.params.step_size"] = trial.suggest_int(
            "scheduler_step_size",
            10, 50
        )
        params["training.scheduler.params.gamma"] = trial.suggest_float(
            "scheduler_gamma",
            0.1, 0.5
        )
    elif params["training.scheduler.name"] == "plateau":
        params["training.scheduler.params.factor"] = trial.suggest_float(
            "scheduler_factor",
            0.1, 0.5
        )
        params["training.scheduler.params.patience"] = trial.suggest_int(
            "scheduler_patience",
            5, 15
        )

    return params