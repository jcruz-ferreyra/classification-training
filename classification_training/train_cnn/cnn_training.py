from datetime import datetime
import logging
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
import pytz
from torch.utils.data import DataLoader

from classification_training.colab import retrieve_and_unzip_data
from classification_training.tracking import log_metrics, log_params, log_tags
from classification_training.training import (
    create_cnn_model,
    create_data_loaders,
    load_class_mappings,
    train_model_loop,
    validate_training_setup,
)

from .types import CNNTrainingContext

logger = logging.getLogger(__name__)


def _create_data_loaders(
    ctx: CNNTrainingContext, class_info: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for a specific trial with modified augmentation."""
    train_loader, val_loader = create_data_loaders(
        dataset_dir=ctx.dataset_dir,
        class_info=class_info,
        batch_size=ctx.training_params["batch_size"],
        input_size=ctx.input_size,
        augmentation_config=ctx.augmentation,
        has_separate_val_split=ctx.validation.get("split_ratio") is None,
        split_ratio=ctx.validation.get("split_ratio", 0.2),
        num_workers=4,
    )

    return train_loader, val_loader


def _load_dataset_info(ctx: CNNTrainingContext) -> Dict[str, Any]:
    """Load dataset information from classification dataset structure and count samples."""
    logger.info(f"Loading dataset info from: {ctx.dataset_dir}")

    try:
        dataset_info = {}
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # Process each split (train, val, test)
        for split in ["train", "val", "test"]:
            split_path = ctx.dataset_dir / split

            if split_path.exists() and split_path.is_dir():
                total_split_samples = 0

                # Count samples per class in this split
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name

                        try:
                            sample_count = len(
                                [
                                    f
                                    for f in class_dir.iterdir()
                                    if f.is_file() and f.suffix.lower() in image_extensions
                                ]
                            )

                            # Store as split_classname format for logging
                            dataset_info[f"{split}_{class_name}"] = sample_count
                            total_split_samples += sample_count

                        except Exception as e:
                            logger.warning(f"Failed to count {split}/{class_name} samples: {e}")
                            dataset_info[f"{split}_{class_name}"] = 0

                # Store split totals
                dataset_info[f"{split}_samples"] = total_split_samples
                logger.debug(f"{split}: {total_split_samples} total samples")

            else:
                logger.warning(f"Split directory not found: {split_path}")
                dataset_info[f"{split}_samples"] = 0

        # Calculate overall dataset info
        total_samples = sum(
            dataset_info.get(f"{split}_samples", 0) for split in ["train", "val", "test"]
        )
        dataset_info["total_samples"] = total_samples

        # Get class names from any available split
        class_names = set()
        for key in dataset_info.keys():
            if "_" in key and not key.endswith("_samples"):
                class_name = key.split("_", 1)[1]  # Get part after first underscore
                class_names.add(class_name)

        dataset_info["class_names"] = sorted(list(class_names))
        dataset_info["num_classes"] = len(class_names)

        logger.info(f"Dataset loaded: {total_samples} total samples, {len(class_names)} classes")

        return dataset_info

    except Exception as e:
        logger.warning(f"Failed to load dataset info from {ctx.dataset_dir}: {e}")
        return {}


def _train_cnn(ctx: CNNTrainingContext, class_info: Dict[str, Any]) -> None:
    """
    Run CNN training with MLflow logging.

    Args:
        ctx: Context containing search configuration
        study: Optuna study object
    """
    logger.info("Starting training")

    # Set MLflow experiment for all trials
    experiment_name = f"{ctx.project_name}"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: {experiment_name}")

    # Train model using shared training loop (SHARED with final training)
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        logger.info(f"Artifacts will be saved to: {ctx.project_dir / run_id}")

        try:
            # Log experiment metadata
            log_tags(
                experiment=ctx.project_name,
                model_family="cnn",
                model_arch=ctx.model_info["architecture"],
                environment=ctx.environment,
                task="classification",
                purpose="training",
                run_id=run_id,
            )

            log_params(
                model_arch=ctx.model_info["architecture"],
                timestamp=datetime.now(pytz.UTC).isoformat(),
                run_id=run_id,
            )

            # Log dataset configuration
            dataset_info = _load_dataset_info(ctx)
            log_params(dataset=ctx.dataset_folder, **dataset_info)

            # Train model
            model = create_cnn_model(model_info=ctx.model_info, checkpoint=ctx.checkpoint_path)

            # Log model configuration
            log_params(
                checkpoint=str(ctx.checkpoint),
                model_params=(
                    sum(p.numel() for p in model.model.parameters())
                    if hasattr(model, "model")
                    else None
                ),
            )

            log_params(
                dropout=ctx.model_info["dropout"], frozen_layers=ctx.model_info["frozen_layers"]
            )
            log_params(**ctx.training_params)
            log_params(**ctx.augmentation)
            log_params(**ctx.early_stopping)

            # Create trial-specific data loaders with modified augmentation
            train_loader, val_loader = _create_data_loaders(ctx, class_info)

            best_val_metrics, history = train_model_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer_name=ctx.training_params["optimizer"],
                lr=ctx.training_params["learning_rate"],
                weight_decay=ctx.training_params["weight_decay"],
                total_epochs=ctx.training_params["epochs"],
                warmup_epochs=ctx.training_params["warmup_epochs"],
                label_smoothing=ctx.training_params["label_smoothing"],
                early_stopping_patience=ctx.early_stopping["patience"],
                early_stopping_monitor=ctx.early_stopping["monitor"],
                early_stopping_min_delta=ctx.early_stopping["min_delta"],
                save_dir=ctx.project_dir / run_id,
            )

            # Log trial results to MLflow as independent run
            log_metrics(split=None, **best_val_metrics)

            # Log model info (without uploading the file)
            log_params(
                best_model=ctx.project_dir / run_id / "weights" / "best.pt",
                artifacts=ctx.project_dir / run_id,
            )

            log_tags(status="completed")

            df_history = pd.DataFrame(history)
            df_history.to_csv(ctx.project_dir / run_id / "result.csv", index=False)

        except Exception as e:
            log_tags(status="failed")
            log_params(error_message=str(e), error_type=type(e).__name__)
            raise


def train_cnn(ctx: CNNTrainingContext) -> None:
    """
    Main function for CNN training.

    Args:
        ctx: Context containing dataset paths and hyperparameter configuration
    """
    logger.info("Starting CNN training")

    if ctx.environment == "colab":
        # Set up dataset in working directory and change dataset dir attribute in ctx
        ctx.data_dir = retrieve_and_unzip_data(ctx.dataset_dir, ctx.dataset_folder)

    # Validate dataset structure and class mappings
    validate_training_setup(
        dataset_dir=ctx.dataset_dir,
        class_mapping_path=ctx.class_mapping_path,
        models_dir=ctx.models_dir,
        has_separate_val_split=ctx.validation.get("split_ratio") is None,
    )

    # Load class mappings and dataset info
    class_info = load_class_mappings(ctx.class_mapping_path)

    # Train the model
    _train_cnn(ctx, class_info)

    logger.info("CNN training completed successfully")
