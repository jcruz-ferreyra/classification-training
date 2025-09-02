import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow

from classification_training.training import (
    create_cnn_model,
    create_data_loaders,
    load_class_mappings,
    eval_model_loop,
)

from .types import CNNEvaluationContext

logger = logging.getLogger(__name__)


def _extract_parameter(
    params: dict, param_name: str, description: str, fallback: str = None
) -> str:
    """Extract a parameter from MLflow run parameters with optional fallback."""
    if param_name not in params:
        if fallback is not None:
            logger.warning(f"Parameter '{param_name}' not found, using fallback: {fallback}")
            return fallback
        raise ValueError(
            f"Required parameter '{param_name}' ({description}) not found in MLflow run"
        )

    value = params[param_name]
    logger.debug(f"Extracted {description}: {value}")
    return value


def _get_mlflow_run_info(ctx: CNNEvaluationContext) -> None:
    """Discover and populate context with latest MLflow run information."""
    logger.info(f"Searching for latest run in experiment: {ctx.experiment}")

    try:
        # Get experiment by name
        experiment = mlflow.get_experiment_by_name(ctx.experiment)
        if experiment is None:
            raise ValueError(f"Experiment '{ctx.experiment}' not found in MLflow")

        # Search for runs in the experiment, ordered by start time (latest first)
        if ctx.run_id is None:
            logger.info(f"No run id provided, using lastest run.")

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )

            if runs.empty:
                raise ValueError(f"No runs found in experiment '{ctx.experiment}'")

            latest_run = runs.iloc[0]
            run_id = latest_run["run_id"]

            logger.info(f"Found latest run: {run_id}")
            logger.debug(f"Run start time: {latest_run['start_time']}")
            logger.debug(f"Run status: {latest_run['status']}")

            # Store run_id in context to log evaluation metrics
            ctx.run_id = run_id

        else:
            logger.info(f"Using provided run id: {ctx.run_id}")

        # Get run details to access parameters
        run_details = mlflow.get_run(run_id)
        params = run_details.data.params

        # Extract required parameters
        if ctx.model_name is None:
            logger.info(f"No model name provided, retrieving from parameters.")
            ctx.model_name = _extract_parameter(
                params, "best_model", "best model checkpoint", "weights/best.pt"
            )
            logger.info(f"Using model checkpoint: {ctx.model_name}")

        logger.info(f"Populated context with run info:")
        logger.info(f"  Dataset folder: {ctx.dataset_folder}")
        logger.info(f"  Data YAML: {ctx.data_yaml}")
        logger.info(f"  Model checkpoint: {ctx.model_path}")
        logger.info(f"  Artifacts dir: {ctx.artifacts_dir}")

    except Exception as e:
        logger.error(f"Failed to discover MLflow run info: {e}")
        raise


def _validate_discovered_paths(ctx: CNNEvaluationContext) -> None:
    """Validate that all discovered paths exist."""
    logger.info("Validating discovered paths")

    if not ctx.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {ctx.model_path}")

    if not ctx.artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {ctx.artifacts_dir}")

    logger.info("All paths validated successfully")


def _load_model(ctx: CNNEvaluationContext) -> Any:
    """Load YOLO model from the discovered model path."""
    logger.info(f"Loading {ctx.eval_config["architecture"]} model from: {ctx.model_path}")

    model_info = {
        "architecture": ctx.eval_config["architecture"],
        "pretrained": ctx.eval_config.get("pretrained", False),
        "num_classes": ctx.eval_config.get("num_classes", 2),
        "dropout": ctx.eval_config.get("dropout", 0.0),
    }

    # Train model
    model = create_cnn_model(model_info=model_info, checkpoint=ctx.model_path)

    return model


def _create_data_loaders(ctx: CNNEvaluationContext, class_info: Dict[str, Any])-> Tuple[DataLoader, DataLoader]:
    """Create data loaders for a specific trial with modified augmentation."""
    test_loader = create_data_loaders(
        dataset_dir=ctx.dataset_dir,
        class_info=class_info,
        batch_size=ctx.eval_config["batch_size"],
        input_size=ctx.eval_config["input_size"],
        augmentation_config=ctx.eval_config,
        num_workers=4,
        eval_mode=True,
    )

    return train_loader, val_loader


def _run_evaluation(ctx):
    # Create trial-specific data loaders with modified augmentation
    train_loader, val_loader = _create_data_loaders(ctx, class_info)


def evaluate_cnn(ctx):
    logger.info("Starting model evaluation")

    # Populates None fields from MLflow
    _get_mlflow_run_info(ctx)

    # Ensure files actually exist
    _validate_discovered_paths(ctx)

    # Load model and store it in context
    model = _load_model(ctx)

    # Continue the same MLflow run for logging evaluation metrics
    with mlflow.start_run(run_id=ctx.run_id):
        map_result, time_result = _run_evaluation(ctx)

        _log_evaluation_metrics(ctx, map_result, time_result)

        _save_results(ctx, map_result, time_result)

    logger.info("Evaluation completed successfully")


