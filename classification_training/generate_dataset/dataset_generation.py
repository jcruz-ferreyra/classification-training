import logging
from pathlib import Path
import random
import shutil
from typing import Dict, List

from .types import DatasetGenerationContext

logger = logging.getLogger(__name__)


def _validate_output_directory(ctx: DatasetGenerationContext):
    """Validate that output directory doesn't already exist."""

    if ctx.output_dir.exists():
        logger.error(f"Directory already exists: {ctx.output_dir}")
        raise ValueError(
            f"Directory already exists: {ctx.output_dir}. "
            "Please remove it before running or choose a different name."
        )

    logger.info("Output directory validation passed")


def _inventory_input_datasets(ctx: DatasetGenerationContext) -> Dict[str, Dict[str, List[str]]]:
    """
    Inventory all input datasets and their class distributions.
    """
    logger.info("Inventorying input datasets and class distributions")

    # Extract unique dataset names from split configuration
    dataset_names = set()
    for split_config in ctx.split_datasets.values():
        dataset_names.update(split_config.keys())

    logger.info(f"Found {len(dataset_names)} unique datasets: {list(dataset_names)}")

    # Get filenames per class per dataset
    dataset_inventory = {}

    for dataset_name in dataset_names:
        logger.info(f"Inventorying dataset: {dataset_name}")

        dataset_path = ctx.input_dir / dataset_name

        # Check if dataset directory exists
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        if not dataset_path.is_dir():
            raise ValueError(f"Dataset path is not a directory: {dataset_path}")

        # Initialize class inventory for this dataset
        class_inventory = {}

        # Inventory each target class
        for target_class in ctx.target_classes:
            class_dir = dataset_path / target_class

            if class_dir.exists() and class_dir.is_dir():
                # Get all image files in this class directory
                image_files = []
                for img_path in class_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        image_files.append(img_path.name)

                class_inventory[target_class] = image_files
                logger.debug(f"  {target_class}: {len(image_files)} images")
            else:
                # Class folder doesn't exist - include empty list
                class_inventory[target_class] = []
                logger.debug(f"  {target_class}: folder not found (0 images)")

        dataset_inventory[dataset_name] = class_inventory

        # Log dataset summary
        total_images = sum(len(files) for files in class_inventory.values())
        logger.info(
            f"Dataset {dataset_name} total: {total_images} images across {len(ctx.target_classes)} classes"
        )

    # Log overall inventory summary
    total_datasets = len(dataset_inventory)
    total_images = sum(
        sum(len(files) for files in class_inv.values()) for class_inv in dataset_inventory.values()
    )
    logger.info(f"Inventory complete: {total_datasets} datasets, {total_images} total images")

    return dataset_inventory


def _flip_split_configuration(split_datasets: dict) -> Dict[str, Dict[str, float]]:
    """Flip split configuration from split -> datasets to dataset -> splits."""
    dataset_proportions = {}

    for split_name, dataset_proportions_dict in split_datasets.items():
        for dataset_name, proportion in dataset_proportions_dict.items():
            if dataset_name not in dataset_proportions:
                dataset_proportions[dataset_name] = {}
            dataset_proportions[dataset_name][split_name] = proportion

    return dataset_proportions


def _allocate_non_overlapping_splits(
    file_paths: List[str], proportions: Dict[str, float], split_allocations: Dict[str, List[str]]
) -> None:
    """Allocate files to splits without overlap (sum <= 1.0)."""
    # Shuffle for random allocation
    shuffled_files = file_paths.copy()
    random.shuffle(shuffled_files)

    start_idx = 0
    for split_name, proportion in proportions.items():
        count = int(len(file_paths) * proportion)
        end_idx = start_idx + count

        allocated_files = shuffled_files[start_idx:end_idx]
        split_allocations[split_name].extend(allocated_files)

        start_idx = end_idx


def _allocate_overlapping_splits(
    file_paths: List[str], proportions: Dict[str, float], split_allocations: Dict[str, List[str]]
) -> None:
    """Allocate files to splits with overlap allowed (sum > 1.0)."""
    # Each split gets its own random sample
    for split_name, proportion in proportions.items():
        count = int(len(file_paths) * proportion)
        allocated_files = random.sample(file_paths, count)
        split_allocations[split_name].extend(allocated_files)


def _generate_splits(
    ctx: DatasetGenerationContext, dataset_inventory: Dict[str, Dict[str, List[str]]]
) -> Dict[str, List[str]]:
    """
    Generate train/val/test splits from dataset inventory.

    Returns:
        Dict mapping split_name -> [dataset_name/target_class/filename, ...]
    """
    logger.info("Generating dataset splits")

    # Flip split_datasets to get dataset -> {split: proportion}
    dataset_proportions = _flip_split_configuration(ctx.split_datasets)

    # Initialize split allocations
    split_allocations = {split: [] for split in ctx.split_datasets.keys()}

    # Process each dataset
    for dataset_name, class_inventory in dataset_inventory.items():
        if dataset_name not in dataset_proportions:
            logger.debug(f"Skipping dataset {dataset_name} (not in split configuration)")
            continue

        proportions = dataset_proportions[dataset_name]
        proportion_sum = sum(proportions.values())

        logger.info(f"Processing dataset {dataset_name} (proportion sum: {proportion_sum:.2f})")

        # Determine allocation strategy based on proportion sum
        overlapping_splits = proportion_sum > 1.0
        if overlapping_splits:
            logger.debug(f"Dataset {dataset_name} will have overlapping splits")

        # Process each target class in this dataset
        for target_class, filenames in class_inventory.items():
            if not filenames:  # Skip empty classes
                continue

            logger.debug(f"  Processing class {target_class}: {len(filenames)} files")

            # Generate file paths with prefix
            file_paths = [f"{dataset_name}/{target_class}/{filename}" for filename in filenames]

            # Allocate files to splits based on proportions
            if proportion_sum <= 1.0:
                # Non-overlapping allocation (sum = 1.0 uses all files, sum < 1.0 leaves some unused)
                _allocate_non_overlapping_splits(file_paths, proportions, split_allocations)
            else:
                # proportion_sum > 1.0 - overlapping splits allowed
                _allocate_overlapping_splits(file_paths, proportions, split_allocations)

    # Log final split summary
    for split_name, file_list in split_allocations.items():
        logger.info(f"Split {split_name}: {len(file_list)} files")

    return split_allocations


def _copy_images(ctx: DatasetGenerationContext, split_allocations: Dict[str, List[str]]) -> None:
    """Copy allocated images to their respective split directories."""
    logger.info("Copying images to split directories")

    for split_name, file_paths in split_allocations.items():
        if not file_paths:
            logger.info(f"No files to copy for split: {split_name}")
            continue

        logger.info(f"Copying {len(file_paths)} files to {split_name} split")

        # Create split directory structure for all target classes
        for target_class in ctx.target_classes:
            split_class_dir = ctx.output_dir / split_name / target_class
            split_class_dir.mkdir(parents=True, exist_ok=True)

        copied_count = 0
        for file_path in file_paths:
            # Parse the file path: dataset_name/target_class/filename
            file_path_obj = Path(file_path)
            path_parts = file_path_obj.parts

            if len(path_parts) != 3:
                logger.warning(f"Invalid file path format: {file_path}")
                continue

            dataset_name, target_class, filename = path_parts

            # Source path
            src_path = ctx.input_dir / dataset_name / target_class / filename

            # Destination path
            dst_path = ctx.output_dir / split_name / target_class / f"{dataset_name}_{filename}"

            try:
                if not src_path.exists():
                    logger.warning(f"Source file not found: {src_path}")
                    continue

                shutil.copy2(str(src_path), str(dst_path))
                copied_count += 1
                logger.debug(f"Copied: {src_path} -> {dst_path}")

            except Exception as e:
                logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
                continue

        logger.info(
            f"Successfully copied {copied_count}/{len(file_paths)} files for {split_name} split"
        )


def _zip_dataset(ctx: DatasetGenerationContext) -> None:
    """Create a compressed zip archive of the generated dataset for drive storage."""
    logger.info(f"Creating zip archive of dataset: {ctx.output_dir}")

    # Create zip file path
    zip_path = ctx.output_dir.parent / f"{ctx.output_dir.name}.zip"

    try:
        import shutil

        # Create zip archive of the entire output directory
        shutil.make_archive(
            base_name=str(zip_path.with_suffix("")),  # Remove .zip extension (added automatically)
            format="zip",
            root_dir=str(ctx.output_dir),
            base_dir=".",
        )

        # Get file sizes for logging
        original_size = sum(f.stat().st_size for f in ctx.output_dir.rglob("*") if f.is_file())
        zip_size = zip_path.stat().st_size
        compression_ratio = (1 - zip_size / original_size) * 100 if original_size > 0 else 0

        logger.info("Dataset archived successfully:")
        logger.info(f"  Archive: {zip_path}")
        logger.info(f"  Original size: {original_size / (1024*1024):.1f} MB")
        logger.info(f"  Compressed size: {zip_size / (1024*1024):.1f} MB")
        logger.info(f"  Compression: {compression_ratio:.1f}%")

    except Exception as e:
        logger.error(f"Failed to create zip archive: {e}")
        raise


def _generate_class_mapping(ctx: DatasetGenerationContext) -> None:
    """Generate class mapping file from target classes."""
    logger.info("Generating class mapping file")

    # Create class to index mapping
    class_mapping = {class_name: idx for idx, class_name in enumerate(ctx.target_classes)}

    # Save mapping to JSON file
    mapping_path = ctx.output_dir / "class_mapping.json"

    try:
        import json

        with open(mapping_path, "w") as f:
            json.dump(class_mapping, f, indent=2)

        logger.info(f"Class mapping saved to: {mapping_path}")
        logger.info(f"Class mappings: {class_mapping}")

    except Exception as e:
        logger.error(f"Failed to save class mapping: {e}")
        raise


def generate_dataset(ctx: DatasetGenerationContext) -> None:
    """
    Generate classification dataset by combining and splitting input image folders.

    Args:
        ctx: DatasetGenerationContext containing input/output paths and configuration
    """
    logger.info("Starting classification dataset generation process")

    # Validate setup and create output structure
    _validate_output_directory(ctx)

    # Load and inventory all available images by dataset and class
    dataset_inventory = _inventory_input_datasets(ctx)

    split_allocations = _generate_splits(ctx, dataset_inventory)

    _copy_images(ctx, split_allocations)

    _generate_class_mapping(ctx)

    # Create archive if using drive storage
    if ctx.output_storage == "drive":
        _zip_dataset(ctx)

    logger.info("Classification dataset generation completed successfully")
