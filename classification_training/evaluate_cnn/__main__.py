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

from classification_training.evaluate_cnn import (
    CNNEvaluationContext,
    evaluate_cnn,
)

logger.info("Starting cnn evaluation pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

required_keys = [
    "dataset_folder",
    "class_mapping",
    "experiment",
    "environment",
    "eval_config",
]
missing_keys = [key for key in required_keys if key not in script_config]
if missing_keys:
    logger.error(f"Missing required config keys: {missing_keys}")
    raise ValueError(f"Missing required config keys: {missing_keys}")

DATASET_FOLDER = script_config["dataset_folder"]
CLASS_MAPPING = script_config["class_mapping"]

EXPERIMENT = script_config["experiment"]
RUN_ID = script_config.get("run_id", None)
MODEL_NAME = script_config.get("model_name", None)

EVAL_CONFIG = script_config["eval_config"]

# Create paths for dataset input (local or drive)
ENVIRONMENT = script_config["environment"]
valid_environments = ["local", "colab"]

if ENVIRONMENT not in valid_environments:
    raise ValueError(f"output_storage configuration should be one of {valid_environments}")
elif ENVIRONMENT == "colab" and (DRIVE_DATA_DIR is None or DRIVE_MODELS_DIR is None):
    raise ValueError(
        "Error accesing Drive directory. Try setting local storage or check provided drive path."
    )

# Select data and models dir based on environment
DATA_DIR = DRIVE_DATA_DIR if ENVIRONMENT == "colab" else DRIVE_DATA_DIR
MODELS_DIR = DRIVE_MODELS_DIR if ENVIRONMENT == "colab" else DRIVE_MODELS_DIR

logger.info(f"Dataset directory: {DATA_DIR}")
logger.info(f"Project directory: {MODELS_DIR}")

# Select initial labelling batch
context = CNNEvaluationContext(
    data_dir=DATA_DIR,
    dataset_folder=DATASET_FOLDER,
    class_mapping=CLASS_MAPPING,
    models_dir=MODELS_DIR,
    experiment=EXPERIMENT,
    run_id=RUN_ID,
    model_name=MODEL_NAME,
    environment=ENVIRONMENT,
    eval_config=EVAL_CONFIG,
)

# Task main function
try:
    start_mlflow(context.environment)
    mlflow.set_tracking_uri(get_mlflow_uri())

    evaluate_cnn(context)

except Exception as e:
    logger.error(e)
    raise

finally:
    stop_mlflow()
