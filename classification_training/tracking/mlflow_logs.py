import logging
import re

import mlflow

logger = logging.getLogger(__name__)


def _sanitize_name(name: str) -> str:
    """Sanitize names for MLflow compatibility."""
    # Keep only: alphanumerics, _, -, ., space, :, /
    sanitized = re.sub(r"[^a-zA-Z0-9_\-\.:/]", "_", str(name))

    # Clean up multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    return sanitized


def _check_metric_is_numeric(value) -> bool:
    """Check if a value can be converted to a numeric type for MLflow metrics."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def log_tags(**tags):
    """Log tags to MLflow with name sanitization."""
    for tag_name, tag_value in tags.items():
        sanitized_name = _sanitize_name(tag_name)
        try:
            mlflow.set_tag(sanitized_name, str(tag_value))
        except Exception as e:
            logger.warning(f"Failed to log tag {sanitized_name}: {e}")


def log_params(**params):
    """Log parameters to MLflow with name sanitization."""
    for param_name, param_value in params.items():
        sanitized_name = _sanitize_name(param_name)
        try:
            mlflow.log_param(sanitized_name, param_value)
        except Exception as e:
            logger.warning(f"Failed to log parameter {sanitized_name}: {e}")


def log_metrics(split: str = "train", **metrics):
    """Log metrics to MLflow with split prefix and validation."""
    for metric_name, metric_value in metrics.items():
        if not _check_metric_is_numeric(metric_value):
            logger.warning(f"Skipping non-numeric metric {metric_name}: {metric_value}")
            continue

        # Add split prefix to metric name
        full_metric_name = f"{split}_{metric_name}"
        sanitized_name = _sanitize_name(full_metric_name)

        try:
            mlflow.log_metric(sanitized_name, float(metric_value))
        except Exception as e:
            logger.warning(f"Failed to log metric {sanitized_name}: {e}")
