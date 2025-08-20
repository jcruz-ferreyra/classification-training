from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CNNHyperparamSearchContext:
    data_dir: Path
    dataset_folder: str
    class_mapping: str

    models_dir: Path
    project_version: str

    model_info: Dict[str, Any]
    training_params: Dict[str, Any]
    augmentation: Dict[str, Any]
    validation: Dict[str, Any]
    early_stopping: Dict[str, Any]

    environment: str

    checkpoint: Optional[str] = None

    def __post_init__(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset_folder

    @property
    def project_name(self) -> str:
        return f"{self.model_info['architecture']}_{self.project_version}"

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
    def checkpoint_path(self) -> Optional[Path]:
        if self.checkpoint is None:
            return None

        return self.models_dir / self.checkpoint

    @property
    def input_size(self) -> int:
        """Get model input size based on architecture."""
        architecture = self.model_info["architecture"]

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
