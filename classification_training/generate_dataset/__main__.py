from pathlib import Path

from classification_training.config import LOCAL_DATA_DIR, DRIVE_DATA_DIR
from classification_training.utils import load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, LOCAL_DATA_DIR)

from classification_training.generate_dataset import (
    DatasetGenerationContext,
    generate_dataset,
)

logger.info("Starting dataset generation pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

required_keys = [
    "input_folder",
    "output_folder",
    "split_datasets",
    "target_classes",
    "output_storage",
]
missing_keys = [key for key in required_keys if key not in script_config]
if missing_keys:
    logger.error(f"Missing required config keys: {missing_keys}")
    raise ValueError(f"Missing required config keys: {missing_keys}")

INPUT_FOLDER = script_config["input_folder"]
OUTPUT_FOLDER = script_config["output_folder"]

SPLIT_DATASETS = script_config["split_datasets"]
TARGET_CLASSES = script_config["target_classes"]

# Create paths for output directory (local or drive)
OUTPUT_STORAGE = script_config["output_storage"]
valid_storages = ["local", "drive"]

if OUTPUT_STORAGE not in valid_storages:
    raise ValueError(f"output_storage configuration should be one of {valid_storages}")
elif OUTPUT_STORAGE == "drive" and DRIVE_DATA_DIR is None:
    raise ValueError(
        "Error accesing Drive directory. Try setting local storage or check provided drive path."
    )

# Select data dir based on storage configuration
OUTPUT_DATA_DIR = DRIVE_DATA_DIR if OUTPUT_STORAGE == "drive" else LOCAL_DATA_DIR

logger.info(f"Input main directory: {LOCAL_DATA_DIR}")
logger.info(f"Output main directory: {OUTPUT_DATA_DIR}")

# Select initial labelling batch
context = DatasetGenerationContext(
    input_data_dir=LOCAL_DATA_DIR,
    input_folder=INPUT_FOLDER,
    output_data_dir=OUTPUT_DATA_DIR,
    output_folder=OUTPUT_FOLDER,
    split_datasets=SPLIT_DATASETS,
    target_classes=TARGET_CLASSES,
    output_storage=OUTPUT_STORAGE,
)

# Task main function
generate_dataset(context)
