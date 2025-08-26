import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def retrieve_and_unzip_data(dataset_dir: Path, dataset_folder: str):
    """Download and extract dataset from drive to colab local storage."""
    logger.info("Starting dataset retrieval and extraction for Colab environment")

    import shutil
    import time
    import zipfile

    dataset_name = dataset_dir.name
    zipfile_name = f"{dataset_name}.zip"

    colab_data_dir = Path("/content/data")
    colab_dataset_dir = colab_data_dir / dataset_folder
    colab_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Copy zip file to colab
    src = dataset_dir.parent / zipfile_name
    dst = colab_dataset_dir.parent / zipfile_name

    # Validate source file exists
    if not src.exists():
        logger.error(f"Source zip file not found: {src}")
        raise FileNotFoundError(f"Dataset zip file not found: {src}")

    try:
        logger.info(f"Copying dataset from drive: {src}")
        start = time.time()
        shutil.copy2(src, dst)
        copy_time = time.time() - start

        # Log copy statistics
        file_size_mb = src.stat().st_size / (1024 * 1024)
        logger.info(
            f"Copied {file_size_mb:.1f} MB in {copy_time:.2f} seconds "
            f"({file_size_mb/copy_time:.1f} MB/s)"
        )

    except Exception as e:
        logger.error(f"Failed to copy dataset zip file: {e}")
        raise

    try:
        logger.info(f"Extracting dataset to: {colab_data_dir}")
        start = time.time()

        with zipfile.ZipFile(dst, "r") as zip_ref:
            zip_ref.extractall(colab_dataset_dir)

        extract_time = time.time() - start

        # Count extracted files
        extracted_files = len(list(colab_data_dir.rglob("*")))
        logger.info(f"Extracted {extracted_files} files in {extract_time:.2f} seconds")

    except zipfile.BadZipFile as e:
        logger.error(f"Corrupted zip file: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract dataset: {e}")
        raise

    try:
        # Clean up zip file to save space
        dst.unlink()
        logger.info("Cleaned up zip file to save space")

    except Exception as e:
        logger.warning(f"Failed to clean up zip file: {e}")

    logger.info("Dataset retrieval and extraction completed successfully")

    # Update context dataset directory
    return colab_data_dir
