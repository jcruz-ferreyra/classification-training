import logging
import os
import subprocess
import time

import mlflow
import requests

from classification_training.config import DRIVE_EXPERIMENTS_DIR, MLFLOW_PORT

logger = logging.getLogger(__name__)


def _check_experiments_dir_not_none():
    if DRIVE_EXPERIMENTS_DIR is None:
        raise ValueError(
            "DRIVE_EXPERIMENTS_DIR is None. Check vavariable definition in .env file and ensure Google Drive is mounted."
        )


def _check_mlflow_running(port=MLFLOW_PORT):
    """Check if MLflow server is already running on the specified port."""
    try:
        response = requests.get(f"http://localhost:{port}", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _wait_for_mlflow_server(timeout=30):
    """Wait for MLflow server to start with timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if _check_mlflow_running():
            return True
        time.sleep(1)

    raise TimeoutError(f"MLflow server failed to start within {timeout} seconds")


def _start_local_mlflow():
    """Start MLflow server for local environment."""
    if _check_mlflow_running():
        logger.info(f"MLflow server already running at http://localhost:{MLFLOW_PORT}")
        return

    logger.info("Starting MLflow server locally...")

    # Ensure the experiments directory exists
    os.makedirs(DRIVE_EXPERIMENTS_DIR, exist_ok=True)

    db_path = os.path.join(DRIVE_EXPERIMENTS_DIR, "mlflow.db")
    logger.info(f"MLflow database path {db_path}")
    artifacts_path = os.path.join(DRIVE_EXPERIMENTS_DIR, "artifacts")
    logger.info(f"MLflow database path {artifacts_path}")

    # Start MLflow server
    subprocess.Popen(
        [
            "mlflow",
            "server",
            "--host",
            "localhost",
            "--port",
            str(MLFLOW_PORT),
            "--backend-store-uri",
            f"sqlite:///{db_path}",
            "--default-artifact-root",
            artifacts_path,
            "--serve-artifacts",
        ]
    )

    # Wait for server to start
    _wait_for_mlflow_server()
    logger.info(f"MLflow server started at http://localhost:{MLFLOW_PORT}")


def _start_colab_mlflow():
    """Start MLflow server for Google Colab environment."""
    if _check_mlflow_running():
        logger.info(f"MLflow server already running at http://localhost:{MLFLOW_PORT}")
        return

    logger.info("Starting MLflow server in Colab...")

    # Ensure the experiments directory exists
    os.makedirs(DRIVE_EXPERIMENTS_DIR, exist_ok=True)

    db_path = os.path.join(DRIVE_EXPERIMENTS_DIR, "mlflow.db")
    logger.info(f"MLflow database path {db_path}")
    artifacts_path = os.path.join(DRIVE_EXPERIMENTS_DIR, "artifacts")
    logger.info(f"MLflow database path {artifacts_path}")

    # Start MLflow server in background
    subprocess.Popen(
        [
            "mlflow",
            "server",
            "--host",
            "localhost",
            "--port",
            str(MLFLOW_PORT),
            "--backend-store-uri",
            f"sqlite:///{db_path}",
            "--default-artifact-root",
            artifacts_path,
            "--dev",  # Development mode for Colab
        ]
    )

    # Wait for server to start
    _wait_for_mlflow_server()
    logger.info(f"MLflow server started at http://localhost:{MLFLOW_PORT}")


def start_mlflow(environment="local"):
    """
    Start MLflow server based on environment.

    Args:
        environment (str): Either "local" or "colab"
    """
    _check_experiments_dir_not_none()

    if environment == "local":
        _start_local_mlflow()
    elif environment == "colab":
        _start_colab_mlflow()
    else:
        raise ValueError("Environment must be either 'local' or 'colab'")


def stop_mlflow():
    """Stop MLflow server (useful for cleanup)."""
    import platform

    try:
        if platform.system() == "Windows":
            # Find PID using the port
            result = subprocess.run(
                f"netstat -ano | findstr :{MLFLOW_PORT}",
                shell=True,
                capture_output=True,
                text=True,
            )

            if result.stdout:
                # Extract PID from netstat output
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if "LISTENING" in line:
                        pid = line.split()[-1]
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True)
                        logger.info(f"MLflow server stopped (PID: {pid})")
                        return

            logger.info("No MLflow server found running")
        else:
            # Linux/Mac: Find PID by port and kill it
            result = subprocess.run(
                f"lsof -ti:{MLFLOW_PORT}", shell=True, capture_output=True, text=True
            )
            if result.stdout.strip():
                pid = result.stdout.strip()
                subprocess.run(f"kill -9 {pid}", shell=True)
                logger.info(f"MLflow server stopped (PID: {pid})")
                return

        logger.info("MLflow server stopped")
    except Exception as e:
        logger.error(f"Could not stop MLflow server: {e}")


def get_mlflow_uri():
    """Get the MLflow tracking URI."""
    return f"http://localhost:{MLFLOW_PORT}"


# Example usage:
if __name__ == "__main__":
    # For local development
    start_mlflow("local")

    # Or for Colab
    # start_mlflow("colab")

    # Set MLflow tracking URI
    import classification_training.tracking.mlflow_setup as mlflow_setup

    mlflow.set_tracking_uri(get_mlflow_uri())

    logger.info(f"MLflow UI available at: {get_mlflow_uri()}")
