from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class CNNTrainingContext:
    data_dir: Path
    dataset_folder: str
    class_mapping: str

    models_dir: Path
    project_name: str

    model: Dict[str, Any]
    training_params: Dict[str, Any]
    augmentation: Dict[str, Any]
    validation: Dict[str, Any]
    early_stopping: Dict[str, Any]

    environment: str

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset_folder

    @property
    def project_dir(self) -> Path:
        return self.models_dir / self.project_name

    @property
    def class_mapping_path(self) -> Path:
        return self.dataset_dir / self.class_mapping

    @property
    def train_dir(self) -> Path:
        return self.dataset_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.dataset_dir / "val"

    @property
    def test_dir(self) -> Path:
        return self.dataset_dir / "test"

    @property
    def input_size(self) -> int:
        """Get model input size based on architecture."""
        architecture = self.model["architecture"]

        # EfficientNet specific sizes
        efficientnet_sizes = {
            "efficientnet_b0": 224,
            "efficientnet_b1": 240,
            "efficientnet_b2": 260,
            "efficientnet_b3": 300,
            "efficientnet_b4": 380,
            "efficientnet_b5": 456,
        }

        if architecture in efficientnet_sizes:
            return efficientnet_sizes[architecture]

        # Default for ResNet and other architectures
        return self.augmentation.get("resize", 224)
