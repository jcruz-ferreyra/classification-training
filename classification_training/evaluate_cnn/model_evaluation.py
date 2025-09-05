import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import mlflow
import pandas as pd
from torch.utils.data import DataLoader

from classification_training.tracking import log_metrics, log_params
from classification_training.training import (
    create_cnn_model,
    create_data_loaders,
    eval_model_loop,
    load_class_mappings,
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
            logger.info("No run id provided, using latest run.")

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
        run_details = mlflow.get_run(ctx.run_id)
        params = run_details.data.params

        # Extract required parameters
        if ctx.model_name is None:
            logger.info(f"No model name provided, retrieving from parameters.")
            ctx.model_name = "weights/best.pt"
            logger.info(f"Using model checkpoint: {ctx.model_name}")

        logger.info("Populated context with run info:")
        logger.info(f"  Model Name: {ctx.model_name}")

    except Exception as e:
        logger.error(f"Failed to discover MLflow run info: {e}")
        raise


def _validate_discovered_paths(ctx: CNNEvaluationContext) -> None:
    """Validate that all discovered paths exist."""
    logger.info("Validating discovered paths")

    logger.error(ctx.models_dir)
    logger.error(ctx.model_path)

    if not ctx.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {ctx.model_path}")

    if not ctx.artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {ctx.artifacts_dir}")

    logger.info("All paths validated successfully")


def _load_model(ctx: CNNEvaluationContext) -> Any:
    """Load YOLO model from the discovered model path."""
    logger.info(f"Loading {ctx.eval_config['architecture']} model from: {ctx.model_path}")

    model_info = {
        "architecture": ctx.eval_config["architecture"],
        "pretrained": ctx.eval_config.get("pretrained", False),
        "num_classes": ctx.eval_config.get("num_classes", 2),
        "dropout": ctx.eval_config.get("dropout", 0.0),
    }

    # Load model from checkpoint
    model = create_cnn_model(model_info=model_info, checkpoint=ctx.model_path)

    return model


def _create_data_loaders(ctx: CNNEvaluationContext, class_info: Dict[str, Any]) -> DataLoader:
    """Create data loaders for a specific trial with modified augmentation."""

    test_loader = create_data_loaders(
        dataset_dir=ctx.dataset_dir,
        class_info=class_info,
        batch_size=ctx.eval_config["batch_size"],
        input_size=ctx.input_size,
        augmentation_config=ctx.eval_config,
        num_workers=4,
        eval_mode=True,
    )

    return test_loader


def _run_evaluation(
    ctx: CNNEvaluationContext, model: Any, class_info: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run model evaluation on test dataset and measure inference timing."""
    test_loader = _create_data_loaders(ctx, class_info)

    start_time = time.time()
    test_metrics = eval_model_loop(
        model=model,
        test_loader=test_loader,
        class_names=class_info["class_names"],
        threshold=ctx.threshold,
    )
    end_time = time.time()

    classification_time = end_time - start_time

    logger.info(
        f"Classification timing results: "
        f"Test dataset inference time: {classification_time:.4f}s"
    )

    return test_metrics, {"total_time": classification_time}


def _log_evaluation_metrics(
    test_results: Dict[str, float],
    time_result: Optional[Dict[str, float]],
) -> None:
    """Log evaluation metrics and timing results to MLflow."""
    logger.info("Logging evaluation metrics to MLflow")

    if test_results is not None:
        log_metrics(split=None, **test_results)

    if test_results is not None:
        log_params(split=None, **test_results)

    if time_result is not None:
        log_metrics(split=None, **time_result)

    logger.info("Evaluation metrics logged successfully")


def _create_flattened_df(eval_results):
    """Flatten everything into a single row (excluding arrays)."""
    required_keys = [
        "test_accuracy",
        "test_f1_macro",
        "test_f1_micro",
        "test_precision_macro",
        "test_recall_macro",
        "test_per_class_f1",
        "test_per_class_precision",
        "test_per_class_recall",
    ]

    for key in required_keys:
        if key not in eval_results:
            raise KeyError(f"Required evaluation result '{key}' not found in results")

    flattened = {
        "test_accuracy": eval_results["test_accuracy"],
        "test_f1_macro": eval_results["test_f1_macro"],
        "test_f1_micro": eval_results["test_f1_micro"],
        "test_precision_macro": eval_results["test_precision_macro"],
        "test_recall_macro": eval_results["test_recall_macro"],
    }

    # Add per-class metrics with prefixed column names
    for class_name, f1_score in eval_results["test_per_class_f1"].items():
        flattened[f"f1_{class_name}"] = f1_score
        flattened[f"precision_{class_name}"] = eval_results["test_per_class_precision"][class_name]
        flattened[f"recall_{class_name}"] = eval_results["test_per_class_recall"][class_name]

    return pd.DataFrame([flattened])


import json


def _save_results(
    ctx: CNNEvaluationContext,
    test_results: Dict[str, Any],
    time_result: Optional[Dict[str, float]],
) -> None:
    """Save evaluation results to CSV and JSON files in artifacts directory."""
    logger.info("Saving evaluation results to CSV and JSON")

    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to pandas DataFrame for CSV export
    results_df = _create_flattened_df(test_results)

    # Add timing results if available
    if time_result is not None:
        for metric_name, metric_value in time_result.items():
            results_df[metric_name] = metric_value

    # Add context information
    results_df["split"] = "test"
    results_df["model_family"] = ctx.eval_config["architecture"]
    results_df["experiment"] = ctx.experiment
    results_df["run_id"] = ctx.run_id

    # Save to CSV
    csv_path = ctx.artifacts_dir / "test_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Flattened results saved to: {csv_path}")

    # Save raw test results to JSON
    json_path = ctx.artifacts_dir / "test_results.json"

    # Combine test_results with timing and context info for complete JSON
    complete_results = {
        "test_metrics": test_results,
        "timing": time_result or {},
        "metadata": {
            "split": "test",
            "model_family": ctx.eval_config["architecture"],
            "experiment": ctx.experiment,
            "run_id": ctx.run_id,
        },
    }

    with open(json_path, "w") as f:
        json.dump(
            complete_results, f, indent=2, default=str
        )  # default=str handles non-serializable types

    logger.info(f"Complete results saved to: {json_path}")


def evaluate_cnn(ctx: CNNEvaluationContext) -> None:
    """Evaluate a trained CNN classification model and log results to MLflow.

    Args:
        ctx: CNNEvaluationContext containing run_id, model paths, dataset paths, and output directory
    """
    logger.info("Starting model evaluation")

    # Populates None fields from MLflow
    _get_mlflow_run_info(ctx)

    # Ensure files actually exist
    _validate_discovered_paths(ctx)

    # Load class mappings and dataset info
    class_info = load_class_mappings(ctx.class_mapping_path)

    # Load model and store it in context
    model = _load_model(ctx)

    # Continue the same MLflow run for logging evaluation metrics
    with mlflow.start_run(run_id=ctx.run_id):
        map_result, time_result = _run_evaluation(ctx, model, class_info)

        _log_evaluation_metrics(map_result, time_result)

        _save_results(ctx, map_result, time_result)

    logger.info("Evaluation completed successfully")
