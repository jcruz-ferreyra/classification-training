from pathlib import Path

import mlflow

from classification_training.config import (
    DRIVE_DATA_DIR,
    DRIVE_MODELS_DIR,
    LOCAL_DATA_DIR,
    LOCAL_MODELS_DIR,
)
from classification_training.tracking import get_mlflow_uri, start_mlflow, stop_mlflow
from classification_training.utils import load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, LOCAL_DATA_DIR)

from classification_training.search_hyperparam_cnn import (
    CNNHyperparamSearchContext,
    search_cnn_hyperparam,
)

logger.info("Starting cnn hypeparameter searching pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

required_keys = [
    "dataset_folder",
    "class_mapping",
    "model_info",
    "project_name",
    "environment",
    "training_params",
    "augmentation",
    "validation",
    "early_stopping",
]
missing_keys = [key for key in required_keys if key not in script_config]
if missing_keys:
    logger.error(f"Missing required config keys: {missing_keys}")
    raise ValueError(f"Missing required config keys: {missing_keys}")

TRAINING_PARAMS = script_config["training_params"]
AUGMENTATION = script_config["augmentation"]
VALIDATION = script_config["validation"]
EARLY_STOPPING = script_config["early_stopping"]

# Create paths for dataset input (local or drive)
ENVIRONMENT = script_config["environment"]
valid_environments = ["local", "colab"]

if ENVIRONMENT not in valid_environments:
    raise ValueError(f"output_storage configuration should be one of {valid_environments}")
elif ENVIRONMENT == "colab" and (DRIVE_DATA_DIR is None or DRIVE_MODELS_DIR is None):
    raise ValueError(
        "Error accesing Drive directory. Try setting local storage or check provided drive path."
    )

DATASET_FOLDER = script_config["dataset_folder"]
CLASS_MAPPING = script_config["class_mapping"]

MODEL_INFO = script_config["model_info"]
PROJECT_NAME = script_config["project_name"]

# Select data and models dir based on environment
DATA_DIR = DRIVE_DATA_DIR if ENVIRONMENT == "colab" else DRIVE_DATA_DIR
MODELS_DIR = DRIVE_MODELS_DIR if ENVIRONMENT == "colab" else DRIVE_MODELS_DIR

logger.info(f"Dataset directory: {DATA_DIR}")
logger.info(f"Project directory: {MODELS_DIR}")

# Select initial labelling batch
context = CNNHyperparamSearchContext(
    data_dir=DATA_DIR,
    dataset_folder=DATASET_FOLDER,
    class_mapping=CLASS_MAPPING,
    models_dir=MODELS_DIR,
    project_name=PROJECT_NAME,
    model_info=MODEL_INFO,
    training_params=TRAINING_PARAMS,
    augmentation=AUGMENTATION,
    validation=VALIDATION,
    early_stopping=EARLY_STOPPING,
    environment=ENVIRONMENT,
)

# Task main function
try:
    start_mlflow(context.environment)
    mlflow.set_tracking_uri(get_mlflow_uri())

    search_cnn_hyperparam(context)

except Exception as e:
    logger.error(e)
    raise

finally:
    stop_mlflow()
