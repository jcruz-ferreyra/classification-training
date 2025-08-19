import logging

import mlflow
import optuna

from classification_training.tracking import log_params

from .types import CNNTrainingContext

logger = logging.getLogger(__name__)


def objective(trial, ctx: CNNTrainingContext) -> float:
    """Optuna objective function for CNN hyperparameter optimization."""

    # Suggest hyperparameters from config ranges
    learning_rate = trial.suggest_float(
        "learning_rate",
        ctx.training_params["learning_rate"][0],
        ctx.training_params["learning_rate"][1],
        log=True,
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        ctx.training_params["weight_decay"][0],
        ctx.training_params["weight_decay"][1],
        log=True,
    )

    dropout = trial.suggest_float("dropout", ctx.model["dropout"][0], ctx.model["dropout"][1])

    optimizer_name = trial.suggest_categorical("optimizer", ctx.training_params["optimizer"])

    warmup_epochs = trial.suggest_int(
        "warmup_epochs",
        ctx.training_params["warmup_epochs"][0],
        ctx.training_params["warmup_epochs"][1],
    )

    random_rotation = trial.suggest_float(
        "random_rotation",
        ctx.augmentation["random_rotation"][0],
        ctx.augmentation["random_rotation"][1],
    )

    brightness = trial.suggest_float(
        "brightness",
        ctx.augmentation["color_jitter"]["brightness"][0],
        ctx.augmentation["color_jitter"]["brightness"][1],
    )

    contrast = trial.suggest_float(
        "contrast",
        ctx.augmentation["color_jitter"]["contrast"][0],
        ctx.augmentation["color_jitter"]["contrast"][1],
    )

    # Create trial-specific config by overriding context values
    trial_config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "optimizer": optimizer_name,
        "warmup_epochs": warmup_epochs,
        "random_rotation": random_rotation,
        "brightness": brightness,
        "contrast": contrast,
    }

    logger.info(f"Trial {trial.number}: {trial_config}")

    try:
        # Train model with suggested hyperparameters
        val_metrics = _train_single_trial(ctx, trial_config)

        # Return the metric to optimize (higher is better for Optuna)
        target_metric = val_metrics[ctx.early_stopping["monitor"]]

        # Log trial results to MLflow
        with mlflow.start_run(nested=True):
            # Log trial hyperparameters
            mlflow.log_params(trial_config)
            mlflow.log_param("trial_number", trial.number)

            # Log trial metrics
            mlflow.log_metrics(val_metrics)
            mlflow.log_metric("target_metric", target_metric)

        logger.info(
            f"Trial {trial.number} completed: {ctx.early_stopping['monitor']} = {target_metric:.4f}"
        )
        return target_metric

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        # Return poor score for failed trials
        return 0.0
