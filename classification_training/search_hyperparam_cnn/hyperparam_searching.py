import copy
import json
import logging
from typing import Any, Dict, Tuple

import mlflow
import optuna
from torch.utils.data import DataLoader

from classification_training.tracking import log_metrics, log_params
from classification_training.training import (
    create_cnn_model,
    create_data_loaders,
    load_class_mappings,
    train_model_loop,
    validate_training_setup,
)

from .types import CNNHyperparamSearchContext

logger = logging.getLogger(__name__)


def _create_optuna_study(ctx: CNNHyperparamSearchContext) -> optuna.Study:
    """
    Create or load Optuna study for hyperparameter optimization.
    """
    logger.info("Setting up Optuna study")

    # Create study name based on project and architecture
    study_name = f"{ctx.project_name}_hyperparam_search"

    # Determine optimization direction based on monitored metric
    monitor_metric = ctx.early_stopping["monitor"]
    if "loss" in monitor_metric.lower():
        direction = "minimize"
    else:
        direction = "maximize"  # accuracy, f1, precision, recall

    logger.info(f"Study optimization direction: {direction} (based on {monitor_metric})")

    # Create storage path for study persistence
    storage_dir = ctx.project_dir / "optuna_studies"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_dir}/{study_name}.db"

    try:
        # Create or load existing study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction=direction,
            load_if_exists=True,  # Resume if study already exists
            sampler=optuna.samplers.TPESampler(seed=42),  # Reproducible sampling
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,  # Don't prune first 5 trials
                n_warmup_steps=10,  # Wait 10 epochs before pruning
                interval_steps=5,  # Check pruning every 2 epochs
            ),
        )

        # Log study info
        logger.info(f"Study created/loaded: {study_name}")
        logger.info(f"Storage: {storage_url}")
        logger.info(f"Number of completed trials: {len(study.trials)}")

        if len(study.trials) > 0:
            logger.info(f"Resuming from {len(study.trials)} previous trials")
            if study.best_trial:
                logger.info(f"Current best value: {study.best_value:.4f}")
                logger.info(f"Current best params: {study.best_params}")

        return study

    except Exception as e:
        logger.error(f"Failed to create Optuna study: {e}")
        raise


def _get_trial_hyperparams(ctx: CNNHyperparamSearchContext, trial: optuna.Trial) -> Dict[str, Any]:
    """Create complete hyperparameter dictionary with trial suggestions and fixed values."""
    hyperparams = {
        "model_info": copy.deepcopy(ctx.model_info),
        "training_params": copy.deepcopy(ctx.training_params),
        "augmentation": copy.deepcopy(ctx.augmentation),
    }

    # Override model parameters with trial suggestions
    hyperparams["model_info"]["dropout"] = trial.suggest_float(
        "dropout", ctx.model_info["dropout"][0], ctx.model_info["dropout"][1]
    )

    architecture = ctx.model_info["architecture"]
    frozen_range = ctx.model_info["frozen_layers"][architecture]
    hyperparams["model_info"]["frozen_layers"] = trial.suggest_int(
        "frozen_layers", frozen_range[0], frozen_range[1]
    )

    # Override training parameters with trial suggestions
    hyperparams["training_params"]["learning_rate"] = trial.suggest_float(
        "learning_rate",
        float(ctx.training_params["learning_rate"][0]),
        float(ctx.training_params["learning_rate"][1]),
        log=True,
    )

    hyperparams["training_params"]["weight_decay"] = trial.suggest_float(
        "weight_decay",
        float(ctx.training_params["weight_decay"][0]),
        float(ctx.training_params["weight_decay"][1]),
        log=True,
    )

    hyperparams["training_params"]["optimizer"] = trial.suggest_categorical(
        "optimizer", ctx.training_params["optimizer"]
    )

    # Override augmentation parameters with trial suggestions
    hyperparams["augmentation"]["random_rotation"] = trial.suggest_float(
        "random_rotation",
        ctx.augmentation["random_rotation"][0],
        ctx.augmentation["random_rotation"][1],
    )

    hyperparams["augmentation"]["color_jitter"]["brightness"] = trial.suggest_float(
        "brightness",
        ctx.augmentation["color_jitter"]["brightness"][0],
        ctx.augmentation["color_jitter"]["brightness"][1],
    )

    hyperparams["augmentation"]["color_jitter"]["contrast"] = trial.suggest_float(
        "contrast",
        ctx.augmentation["color_jitter"]["contrast"][0],
        ctx.augmentation["color_jitter"]["contrast"][1],
    )

    return hyperparams


def _create_trial_data_loaders(
    ctx: CNNHyperparamSearchContext, trial_hyperparams: Dict[str, Any], class_info: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for a specific trial with modified augmentation."""
    train_loader, val_loader = create_data_loaders(
        dataset_dir=ctx.dataset_dir,
        class_info=class_info,
        batch_size=ctx.training_params["batch_size"],
        input_size=ctx.input_size,
        augmentation_config=trial_hyperparams["augmentation"],
        has_separate_val_split=ctx.validation.get("split_ratio") is None,
        split_ratio=ctx.validation.get("split_ratio", 0.2),
        num_workers=4,
    )

    return train_loader, val_loader


def _train_trial_model(
    ctx: CNNHyperparamSearchContext,
    trial_hyperparams: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    trial: optuna.Trial,
) -> Dict[str, float]:
    """Train model for a single trial and return validation metrics."""

    # Create model with trial-specific parameters
    model = create_cnn_model(
        model_info=trial_hyperparams["model_info"], checkpoint=ctx.checkpoint_path
    )

    # Train model using shared training loop (SHARED with final training)
    best_val_metrics, _ = train_model_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_name=trial_hyperparams["training_params"]["optimizer"],
        lr=trial_hyperparams["training_params"]["learning_rate"],
        weight_decay=trial_hyperparams["training_params"]["weight_decay"],
        total_epochs=trial_hyperparams["training_params"]["epochs"],
        warmup_epochs=trial_hyperparams["training_params"]["warmup_epochs"],
        label_smoothing=trial_hyperparams["training_params"]["label_smoothing"],
        early_stopping_patience=ctx.early_stopping["patience"],
        early_stopping_monitor=ctx.early_stopping["monitor"],
        early_stopping_min_delta=ctx.early_stopping["min_delta"],
        trial=trial,  # For pruning support
    )

    return best_val_metrics


def _log_trial_to_mlflow(
    trial: optuna.Trial,
    trial_hyperparams: Dict[str, Any],
    val_metrics: Dict[str, float],
    target_metric: float,
) -> None:
    """Log individual trial results to MLflow as nested run."""

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        # Log trial hyperparameters
        log_params(**trial_hyperparams)
        log_params(trial_number=trial.number)

        # Log trial metrics
        log_metrics(split=None, **val_metrics)
        log_metrics(split=None, target_metric=target_metric)


def _run_optimization_trials(
    ctx: CNNHyperparamSearchContext, study: optuna.Study, class_info: Dict[str, Any]
) -> None:
    """
    Run Optuna optimization trials for hyperparameter search.

    Args:
        ctx: Context containing search configuration
        study: Optuna study object
    """
    logger.info("Starting hyperparameter optimization trials")

    # Set MLflow experiment for all trials
    experiment_name = f"{ctx.project_name}_hyperparameter_search"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: {experiment_name}")

    # Get optimization configuration
    n_trials = ctx.training_params.get("n_trials", 50)
    trial_epochs = ctx.training_params.get("epochs", 15)

    logger.info(f"Running {n_trials} trials with {trial_epochs} epochs each")

    def objective(trial):
        """Objective function for single trial optimization."""
        logger.info(f"Starting trial {trial.number}")

        try:
            # Extract hyperparameters from trial
            trial_hyperparams = _get_trial_hyperparams(ctx, trial)

            # Create trial-specific data loaders with modified augmentation
            trial_train_loader, trial_val_loader = _create_trial_data_loaders(
                ctx, trial_hyperparams, class_info
            )

            # Train model with trial hyperparameters
            val_metrics = _train_trial_model(
                ctx, trial_hyperparams, trial_train_loader, trial_val_loader, trial
            )

            # Get target metric for optimization
            target_metric = val_metrics[ctx.early_stopping["monitor"]]

            # Log trial results to MLflow as independent run
            _log_trial_to_mlflow(trial, trial_hyperparams, val_metrics, target_metric)

            logger.info(
                f"Trial {trial.number} completed: {ctx.early_stopping['monitor']} = {target_metric:.4f}"
            )
            return target_metric

        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} pruned")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return poor score for failed trials
            if "loss" in ctx.early_stopping["monitor"].lower():
                return float("inf")
            else:
                return 0.0

    # Run optimization (no parent MLflow run)
    try:
        study.optimize(objective, n_trials=n_trials)

        # Log study completion
        logger.info(f"Optimization completed after {len(study.trials)} trials")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        logger.info(f"Completed {len(study.trials)} trials before interruption")
        if study.best_trial:
            logger.info(f"Best value so far: {study.best_value:.4f}")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise


def _save_best_hyperparams(ctx: CNNHyperparamSearchContext, study: optuna.Study) -> None:
    """
    Extract and save best hyperparameters from completed Optuna study.

    Args:
        ctx: Context containing project configuration
        study: Completed Optuna study

    Returns:
        Dictionary containing best hyperparameters
    """
    logger.info("Extracting and saving best hyperparameters from study")

    if study.best_trial is None:
        raise ValueError("No completed trials found in study")

    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = study.best_value

    logger.info(f"Best trial #{best_trial.number}")
    logger.info(f"Best {ctx.early_stopping['monitor']}: {best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")

    # Create comprehensive results
    optimization_results = {
        "study_info": {
            "best_trial_number": best_trial.number,
            "best_value": best_value,
            "monitored_metric": ctx.early_stopping["monitor"],
            "total_trials": len(study.trials),
            "completed_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "pruned_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
            "failed_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            ),
        },
        "best_hyperparams": best_params,
        "architecture": ctx.model_info["architecture"],
        "experiment": ctx.project_name,
    }

    # Save to JSON file
    results_path = ctx.project_dir / "optuna_studies" / "best_hyperparams.json"

    try:
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(optimization_results, f, indent=2)

        logger.info(f"Best hyperparameters saved to: {results_path}")

    except Exception as e:
        logger.error(f"Failed to save best hyperparameters: {e}")
        raise


def search_cnn_hyperparam(ctx: CNNHyperparamSearchContext) -> None:
    """
    Main function for CNN hyperparameter optimization using Optuna.

    Args:
        ctx: Context containing dataset paths and hyperparameter search configuration
    """
    logger.info("Starting CNN hyperparameter search")

    # Validate dataset structure and class mappings
    validate_training_setup(
        dataset_dir=ctx.dataset_dir,
        class_mapping_path=ctx.class_mapping_path,
        models_dir=ctx.models_dir,
        has_separate_val_split=ctx.validation.get("split_ratio") is None,
    )

    # Load class mappings and dataset info
    class_info = load_class_mappings(ctx.class_mapping_path)

    # Setup Optuna study for hyperparameter optimization
    study = _create_optuna_study(ctx)

    # Run hyperparameter optimization trials
    _run_optimization_trials(ctx, study, class_info)

    # Extract and save best hyperparam
    _save_best_hyperparams(ctx, study)

    logger.info("CNN hyperparameter search completed successfully")
