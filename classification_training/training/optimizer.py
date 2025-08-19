import logging
from typing import Any, Dict

import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_optimizer(model: nn.Module, optimizer, lr, weight_decay) -> optim.Optimizer:
    """Create optimizer with trial-specific parameters."""

    logger.info(f"Creating {optimizer} optimizer")
    logger.info(f"Learning rate: {lr:.2e}, Weight decay: {weight_decay:.2e}")

    if optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
