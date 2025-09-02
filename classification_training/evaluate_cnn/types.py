from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CNNEvaluationContext:
    data_dir: Path
    dataset_folder: str
    class_mapping: str

    models_dir: Path
    experiment: str
    run_id: Optional[str]
    model_name: Optional[str]

    environment: str

    eval_config: Dict[str, Any]

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset_folder

    @property
    def class_mapping_path(self) -> Path:
        return self.dataset_dir / self.class_mapping

    @property
    def model_path(self) -> Optional[Path]:
        if self.run_id is None or self.model_name is None:
            return None
        else:
            return self.models_dir / self.experiment / self.run_id / self.model_name

    @property
    def artifacts_dir(self) -> Optional[Path]:
        if self.run_id is None:
            return None
        else:
            return self.models_dir / self.experiment / self.run_id

    @property
    def test_dir(self) -> Path:
        return self.dataset_dir / "test"

    @property
    def input_size(self) -> int:
        """Get model input size based on architecture."""
        architecture = self.eval_config["architecture"]

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
        return self.eval_config.get("resize", 224)

    def __post_init__(self):
        if not self.test_dir.exists():
            raise ValueError(f"Test directory does not exist: {self.test_dir}")
