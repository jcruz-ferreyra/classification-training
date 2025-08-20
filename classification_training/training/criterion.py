from collections import Counter
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def create_criterion(
    label_smoothing: float,
    train_loader: Optional[DataLoader] = None,
    use_class_weights: bool = True,
    device: str = "cpu",
) -> nn.Module:
    """
    Create loss function with optional class weights for imbalance handling.

    Args:
        label_smoothing: Label smoothing value
        train_loader: Training data loader (for computing class weights)
        use_class_weights: Whether to compute and use class weights

    Returns:
        Configured loss function
    """
    logger.info(f"Creating CrossEntropyLoss with label smoothing: {label_smoothing:.3f}")

    class_weights = None

    if use_class_weights and train_loader is not None:
        logger.info("Computing class weights from training data")
        class_weights = _compute_class_weights(train_loader)
        logger.info(f"Class weights: {class_weights}")

        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)

    return criterion


def _compute_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """Compute inverse frequency class weights from training data."""

    # Collect all labels from the dataset
    all_labels = []

    logger.debug("Scanning training dataset to compute class weights")

    # Handle different dataset types
    dataset = train_loader.dataset
    if hasattr(dataset, "dataset"):  # Subset case
        dataset = dataset.dataset

    if hasattr(dataset, "targets"):
        # ImageFolder has targets attribute
        all_labels = dataset.targets
    else:
        # Manual extraction if targets not available
        for _, target in train_loader:
            if isinstance(target, torch.Tensor):
                all_labels.extend(target.tolist())
            else:
                all_labels.extend(target)

    # Count class frequencies
    class_counts = Counter(all_labels)
    num_classes = len(class_counts)
    total_samples = len(all_labels)

    logger.info(f"Class distribution: {dict(class_counts)}")

    # Calculate inverse frequency weights
    class_weights = torch.zeros(num_classes)
    for class_idx, count in class_counts.items():
        # Inverse frequency: total_samples / (num_classes * class_count)
        weight = total_samples / (num_classes * count)
        class_weights[class_idx] = weight

    # Normalize weights so they sum to num_classes
    class_weights = class_weights / class_weights.mean()

    logger.info(f"Computed class weights: {class_weights.tolist()}")

    return class_weights
