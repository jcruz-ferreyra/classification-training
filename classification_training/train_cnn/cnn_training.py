import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def train_cnn(ctx: CNNTrainingContext) -> None:
    """
    Train CNN model for demographic classification with optional hyperparameter optimization.

    Args:
        ctx: CNNTrainingContext containing all training configuration
    """
    logger.info("Starting CNN training pipeline")

    # Validate dataset structure and class mappings
    _validate_training_setup(ctx)

    # Load class mappings and dataset statistics
    class_info = _load_class_mappings(ctx)

    # Create data loaders for training and validation
    train_loader, val_loader = _create_data_loaders(ctx)

    # Perform hyperparameter optimization if configured
    if _should_run_hyperparameter_search(ctx):
        best_hyperparams = _run_hyperparameter_optimization(ctx, train_loader, val_loader)
        _update_context_with_best_params(ctx, best_hyperparams)

    # Train final model with optimized or default hyperparameters
    model, training_metrics = _train_final_model(ctx, train_loader, val_loader)

    # Evaluate model performance on test set
    test_metrics = _evaluate_final_model(ctx, model)

    # Save model artifacts and results
    _save_model_and_artifacts(ctx, model, training_metrics, test_metrics)

    # Log final results to MLflow
    _log_final_results(ctx, training_metrics, test_metrics)

    logger.info("CNN training pipeline completed successfully")
