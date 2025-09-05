import logging
from pathlib import Path
from typing import Any, Dict, Optional

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _freeze_model_layers(model, num_frozen_layers: int, architecture: str) -> None:
    """Freeze specified number of layers based on architecture."""
    if num_frozen_layers == 0:
        logger.debug("No layers to freeze")
        return

    logger.info(f"Freezing {num_frozen_layers} layers for {architecture}")

    frozen_params = 0
    total_params = sum(p.numel() for p in model.parameters())

    if "efficientnet" in architecture:
        # Freeze feature extraction blocks
        blocks = list(model.blocks)
        available_blocks = len(blocks)
        layers_to_freeze = min(num_frozen_layers, available_blocks)

        logger.debug(f"EfficientNet has {available_blocks} blocks, freezing {layers_to_freeze}")

        for i in range(min(num_frozen_layers, len(blocks))):
            for param in blocks[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()

    elif "resnet" in architecture:
        # Freeze layer groups (conv1, bn1, layer1, layer2, etc.)
        layer_groups = [
            model.conv1,
            model.bn1,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]
        available_groups = len(layer_groups)
        groups_to_freeze = min(num_frozen_layers, available_groups)

        logger.debug(f"ResNet has {available_groups} layer groups, freezing {groups_to_freeze}")

        for i in range(min(num_frozen_layers, len(layer_groups))):
            for param in layer_groups[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()

    else:
        logger.warning(f"Unknown architecture for freezing: {architecture}")
        return

    frozen_percentage = (frozen_params / total_params) * 100
    logger.info(f"Frozen {frozen_params:,}/{total_params:,} parameters ({frozen_percentage:.1f}%)")


def create_cnn_model(model_info: Dict[str, Any], checkpoint: Optional[Path] = None) -> nn.Module:
    """Create model with trial-specific parameters."""
    logger.info(f"Creating {model_info['architecture']} model")

    # Create base model
    model = timm.create_model(
        model_info["architecture"],
        pretrained=model_info["pretrained"],
        num_classes=model_info["num_classes"],
        drop_rate=model_info["dropout"],
    )

    # Load custom checkpoint if specified
    if checkpoint is not None:
        logger.info(f"Loading custom checkpoint from: {checkpoint}")

        try:
            # Use weights_only=False for trusted checkpoints
            checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint_data["model_state_dict"])
            logger.info("Custom checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint}: {e}")
            raise

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Dropout rate: {model_info['dropout']:.3f}")

    # Apply layer freezing based on trial hyperparameters
    if "frozen_layers" in model_info:
        _freeze_model_layers(model, model_info["frozen_layers"], model_info["architecture"])

        final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters after freezing: {final_trainable:,}")
    else:
        logger.info("No layer freezing specified")

    final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters after freezing: {final_trainable:,}")

    return model
