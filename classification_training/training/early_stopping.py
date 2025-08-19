import logging
from typing import Dict


logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility class."""

    def __init__(self, patience: int, monitor: str, min_delta: float = 0.0):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            monitor: Metric to monitor (e.g., 'val_accuracy', 'val_loss')
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

        # Determine if higher is better based on metric name
        self.higher_is_better = "loss" not in monitor.lower()

    def __call__(self, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop.

        Args:
            metrics: Dictionary of current epoch metrics

        Returns:
            True if training should stop, False otherwise
        """
        if self.monitor not in metrics:
            logger.warning(f"Monitored metric '{self.monitor}' not found in metrics")
            return False

        current_score = metrics[self.monitor]

        if self.best_score is None:
            self.best_score = current_score
            logger.debug(f"Initial {self.monitor}: {current_score:.4f}")
            return False

        # Check for improvement
        if self.higher_is_better:
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            logger.debug(f"Improvement in {self.monitor}: {current_score:.4f}")
        else:
            self.counter += 1
            logger.debug(
                f"No improvement in {self.monitor}: {current_score:.4f} (patience: {self.counter}/{self.patience})"
            )

        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(
                f"Early stopping triggered after {self.patience} epochs without improvement"
            )
            return True

        return False
