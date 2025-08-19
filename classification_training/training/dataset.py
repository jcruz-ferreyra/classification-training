import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)


def validate_training_setup(
    dataset_dir: Path,
    class_mapping_path: Path,
    models_dir: Path,
    has_separate_val_split: bool = True,
) -> None:
    """
    Validate that all required paths and files exist for training.

    Args:
        dataset_dir: Path to dataset directory
        class_mapping_path: Path to class mapping JSON file
        models_dir: Path to models directory for saving
        has_separate_val_split: Whether validation split exists as separate folder
    """
    logger.info("Validating training setup")

    # Validate dataset directory exists
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Validate split directories exist
    required_splits = ["train"]
    if has_separate_val_split:
        required_splits.append("val")

    for split in required_splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Validate class mapping file exists
    if not class_mapping_path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")

    # Validate models directory exists (for saving)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Validate that training directory has class subdirectories
    train_dir = dataset_dir / "train"
    train_class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not train_class_dirs:
        raise ValueError(f"No class directories found in training directory: {train_dir}")

    logger.info(f"Found {len(train_class_dirs)} classes in training directory")

    # Log basic dataset info
    for class_dir in train_class_dirs:
        image_count = len(
            [f for f in class_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
        logger.debug(f"Class {class_dir.name}: {image_count} images")

    logger.info("Training setup validation completed")


def load_class_mappings(class_mapping_path: Path) -> Dict[str, Any]:
    """
    Load class mappings and return class information for training.

    Args:
        class_mapping_path: Path to class mapping JSON file

    Returns:
        Dict containing class_to_idx, idx_to_class, and num_classes
    """
    logger.info(f"Loading class mappings from: {class_mapping_path}")

    try:
        import json

        with open(class_mapping_path, "r") as f:
            class_to_idx = json.load(f)

        # Create reverse mapping
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

        # Get number of classes
        num_classes = len(class_to_idx)

        logger.info(f"Loaded {num_classes} classes: {list(class_to_idx.keys())}")

        class_info = {
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "num_classes": num_classes,
            "class_names": list(class_to_idx.keys()),
        }

        return class_info

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in class mapping file: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Class mapping file not found: {class_mapping_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load class mappings: {e}")
        raise


def create_transforms(
    input_size: int, augmentation_config: Dict[str, Any], is_training: bool = True
) -> transforms.Compose:
    """
    Create training or validation transforms based on configuration.

    Args:
        input_size: Target image size for model input
        augmentation_config: Augmentation configuration dict
        is_training: Whether to apply training augmentations

    Returns:
        Composed transforms for the dataset
    """
    transform_list = []

    # Resize and crop
    transform_list.extend(
        [
            transforms.Resize(input_size + 32),  # Slightly larger for random crop
            (
                transforms.CenterCrop(input_size)
                if not is_training
                else transforms.RandomCrop(input_size)
            ),
        ]
    )

    if is_training:
        # Training augmentations

        # Random horizontal flip
        flip_prob = augmentation_config.get("random_flip", 0.5)
        if flip_prob > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))

        # Random rotation
        rotation_degrees = augmentation_config.get("random_rotation", 0)
        if rotation_degrees > 0:
            transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

        # Gaussian blur
        blur_prob = augmentation_config.get("blur_probability", 0.0)
        if blur_prob > 0:
            transform_list.append(
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=blur_prob)
            )

        # Color jitter
        color_jitter = augmentation_config.get("color_jitter", {})
        if any(color_jitter.values()):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter.get("brightness", 0.0),
                    contrast=color_jitter.get("contrast", 0.0),
                    saturation=color_jitter.get("saturation", 0.0),
                    hue=color_jitter.get("hue", 0.0),
                )
            )

    # Convert to tensor and normalize (always applied)
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=augmentation_config["normalize"]["mean"],
                std=augmentation_config["normalize"]["std"],
            ),
        ]
    )

    return transforms.Compose(transform_list)


def create_data_loaders(
    dataset_dir: Path,
    class_info: Dict[str, Any],
    batch_size: int,
    input_size: int,
    augmentation_config: Dict[str, Any],
    has_separate_val_split: bool = True,
    split_ratio: float = 0.2,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        dataset_dir: Path to dataset directory
        class_info: Class information dict from load_class_mappings
        batch_size: Batch size for data loaders
        input_size: Target input size for model
        augmentation_config: Augmentation configuration
        has_separate_val_split: Whether val folder exists separately
        split_ratio: Train/val split ratio if no separate val folder
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating data loaders")

    # Create transforms
    train_transforms = create_transforms(input_size, augmentation_config, is_training=True)
    val_transforms = create_transforms(input_size, augmentation_config, is_training=False)

    # Create datasets without forcing mapping
    if has_separate_val_split:
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"

        train_dataset = ImageFolder(root=str(train_dir), transform=train_transforms)
        val_dataset = ImageFolder(root=str(val_dir), transform=val_transforms)

        logger.info(
            f"Loaded separate splits: {len(train_dataset)} train, {len(val_dataset)} val samples"
        )

    else:
        # Split train directory into train/val
        train_dir = dataset_dir / "train"

        base_dataset = ImageFolder(root=str(train_dir), transform=None)

        # Calculate split sizes
        total_size = len(base_dataset)
        val_size = int(total_size * split_ratio)
        train_size = total_size - val_size

        # Split dataset
        train_indices, val_indices = random_split(
            range(total_size), [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # Create datasets with appropriate transforms
        train_dataset = torch.utils.data.Subset(
            ImageFolder(root=str(train_dir), transform=train_transforms), train_indices
        )
        val_dataset = torch.utils.data.Subset(
            ImageFolder(root=str(train_dir), transform=val_transforms), val_indices
        )

        logger.info(
            f"Split train directory: {len(train_dataset)} train, {len(val_dataset)} val samples"
        )

    # Validate that JSON mapping matches PyTorch's alphabetical order
    active_dataset = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
    pytorch_class_to_idx = active_dataset.class_to_idx
    expected_class_to_idx = class_info["class_to_idx"]

    # Check if mappings match exactly
    for class_name, expected_idx in expected_class_to_idx.items():
        pytorch_idx = pytorch_class_to_idx.get(class_name)
        if pytorch_idx != expected_idx:
            raise ValueError(
                f"Class mapping mismatch for '{class_name}': "
                f"JSON file has index {expected_idx}, but PyTorch assigned index {pytorch_idx}. "
                f"Expected JSON mapping: {dict(sorted(pytorch_class_to_idx.items()))} "
                f"(alphabetical order)"
            )

    logger.info("Class mapping validation passed - JSON matches PyTorch alphabetical order")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Created data loaders with batch size {batch_size}")
    return train_loader, val_loader
