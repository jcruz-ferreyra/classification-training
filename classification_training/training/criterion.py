from typing import Any, Dict

import torch.nn as nn


def create_criterion(label_smoothing: float) -> nn.Module:
    """Create loss function with trial-specific parameters."""
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
