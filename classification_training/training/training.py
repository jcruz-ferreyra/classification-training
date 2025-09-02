import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .criterion import create_criterion
from .early_stopping import EarlyStopping
from .optimizer import create_optimizer
from .scheduler import create_cosine_scheduler

logger = logging.getLogger(__name__)


def _train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    # Create progress bar for training batches
    train_pbar = tqdm(train_loader, desc="Training", leave=False)

    for data, target in train_pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

        # Update progress bar with current metrics
        current_accuracy = correct / total if total > 0 else 0.0
        train_pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{current_accuracy:.4f}"})

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return {"train_loss": avg_loss, "train_accuracy": accuracy}


def _validate_epoch(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str
) -> Dict[str, float]:
    """Validate model for one epoch."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calculate basic metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_preds)

    # Calculate sklearn metrics
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    f1_micro = f1_score(
        all_targets, all_preds, average="micro"
    )  # Same as accuracy for multi-class
    precision_macro = precision_score(all_targets, all_preds, average="macro")
    recall_macro = recall_score(all_targets, all_preds, average="macro")

    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_f1_macro": f1_macro,
        "val_f1_micro": f1_micro,
        "val_precision_macro": precision_macro,
        "val_recall_macro": recall_macro,
    }


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Path,
) -> None:
    """Save complete model checkpoint for resuming training."""
    # Ensure parent directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    try:
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
        raise


def train_model_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    total_epochs: int,
    warmup_epochs: int,
    label_smoothing: float,
    early_stopping_patience: int,
    early_stopping_monitor: str,
    early_stopping_min_delta: float,
    device: Optional[str] = None,
    trial: Optional[optuna.Trial] = None,
    metrics_callback: Optional[Callable] = None,
    save_dir: Optional[Path] = None,
    save_frequency: int = 1,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Shared training loop for CNN models.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer_name: Name of optimizer (adamw, adam, sgd)
        lr: Learning rate
        weight_decay: Weight decay value
        total_epochs: Total number of epochs to train
        warmup_epochs: Number of warmup epochs
        label_smoothing: Label smoothing value for loss function
        early_stopping_patience: Patience for early stopping
        early_stopping_monitor: Metric to monitor for early stopping
        early_stopping_min_delta: Minimum delta for improvement
        device: Device to train on (auto-detect if None)
        trial: Optuna trial for pruning (optional)
        metrics_callback: Optional callback for custom metrics logging
        save_dir: Directory to save models (optional, saves to save_dir/weights/)
        save_frequency: Frequency to save model checkpoints (in epochs)

    Returns:
        Dict of final validation metrics
    """
    logger.info(f"Starting training loop for {total_epochs} epochs")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    logger.info(f"Training on device: {device}")

    # Create save directory if specified
    if save_dir is not None:
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model checkpoints will be saved to: {weights_dir}")

    # Initialize optimizer and scheduler
    optimizer = create_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = create_cosine_scheduler(optimizer, warmup_epochs, total_epochs)

    # Initialize loss function
    criterion = create_criterion(
        label_smoothing, train_loader, use_class_weights=True, device=device
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        monitor=early_stopping_monitor,
        min_delta=early_stopping_min_delta,
    )

    best_val_metrics = {}
    best_epoch = 0

    history = []

    for epoch in range(total_epochs):
        logger.info(f"Epoch {epoch+1}/{total_epochs}")

        # Training phase
        train_metrics = _train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation phase
        val_metrics = _validate_epoch(model, val_loader, criterion, device)

        # Scheduler step
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            logger.debug(f"Learning rate: {current_lr:.2e}")

        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics}
        epoch_record = {"epoch": epoch + 1, **epoch_metrics}
        history.append(epoch_record)

        # Log epoch results
        logger.info(
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
            f"Val F1: {val_metrics['val_f1_macro']:.4f}"
        )

        # Custom metrics callback (for MLflow logging, etc.)
        if metrics_callback:
            metrics_callback(epoch, epoch_metrics)

        # Optuna pruning check
        if trial is not None:
            # Report intermediate value for pruning
            trial.report(val_metrics[early_stopping_monitor], epoch)

            # Check if trial should be pruned
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()

        # Update best metrics and save best model
        monitor_value = val_metrics[early_stopping_monitor]

        if early_stopping.higher_is_better:
            best_monitor_value = best_val_metrics.get(early_stopping_monitor, -float("inf"))
            improved = monitor_value > best_monitor_value
        else:
            best_monitor_value = best_val_metrics.get(early_stopping_monitor, float("inf"))
            improved = monitor_value < best_monitor_value

        if improved:
            best_val_metrics = val_metrics.copy()
            best_epoch = epoch + 1

            # Save best model checkpoint
            if save_dir is not None:
                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    checkpoint_path=weights_dir / "best.pt",
                )

        # Save last model checkpoint
        if save_dir is not None and (epoch + 1) % save_frequency == 0:
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics=val_metrics,
                checkpoint_path=weights_dir / "last.pt",
            )

        # Early stopping check
        if early_stopping(val_metrics):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    if save_dir is not None and best_epoch > 0:
        logger.info(f"Best model saved from epoch {best_epoch}")

    # Save last model checkpoint
    if save_dir is not None:
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            metrics=val_metrics,
            checkpoint_path=weights_dir / "last.pt",
        )

    logger.info("Training loop completed")
    return best_val_metrics, history


def eval_model_loop(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluation loop for CNN models on test dataset.

    Args:
        model: Trained PyTorch model to evaluate
        test_loader: Test data loader
        class_names: List of class names for reporting
        device: Device to run evaluation on (auto-detect if None)

    Returns:
        Dict containing evaluation metrics and detailed results
    """
    logger.info("Starting model evaluation")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()
    logger.info(f"Evaluating on device: {device}")

    all_preds = []
    all_targets = []
    all_probs = []

    # Run evaluation
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Evaluating", leave=False)

        for data, target in test_pbar:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)

            # Collect results
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Update progress bar
            batch_accuracy = (pred == target).float().mean().item()
            test_pbar.set_postfix({"Acc": f"{batch_accuracy:.4f}"})

    # Calculate metrics
    accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    f1_micro = f1_score(all_targets, all_preds, average="micro")
    precision_macro = precision_score(all_targets, all_preds, average="macro")
    recall_macro = recall_score(all_targets, all_preds, average="macro")

    # Per-class metrics
    f1_per_class = f1_score(all_targets, all_preds, average=None)
    precision_per_class = precision_score(all_targets, all_preds, average=None)
    recall_per_class = recall_score(all_targets, all_preds, average=None)

    # Create detailed classification report
    class_report = classification_report(
        all_targets, all_preds, target_names=class_names, output_dict=True
    )

    # Compile results
    eval_results = {
        # Overall metrics
        "test_accuracy": accuracy,
        "test_f1_macro": f1_macro,
        "test_f1_micro": f1_micro,
        "test_precision_macro": precision_macro,
        "test_recall_macro": recall_macro,
        # Per-class metrics
        "per_class_f1": dict(zip(class_names, f1_per_class)),
        "per_class_precision": dict(zip(class_names, precision_per_class)),
        "per_class_recall": dict(zip(class_names, recall_per_class)),
        # Raw data for further analysis
        "predictions": all_preds,
        "targets": all_targets,
        "probabilities": all_probs,
        # Detailed report
        "classification_report": class_report,
        # Dataset info
        "total_samples": len(all_targets),
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    # Log summary results
    logger.info("Evaluation completed")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 (macro): {f1_macro:.4f}")
    logger.info(f"Test F1 (micro): {f1_micro:.4f}")

    # Log per-class results
    for i, class_name in enumerate(class_names):
        logger.info(
            f"{class_name}: F1={f1_per_class[i]:.4f}, Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}"
        )

    return eval_results
