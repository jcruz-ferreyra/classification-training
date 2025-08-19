from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DatasetGenerationContext:
    input_data_dir: Path
    input_folder: str
    output_data_dir: Path
    output_folder: str

    split_datasets: Dict[str, List[str]]
    target_classes: List[str]

    output_storage: str = "local"

    def __post_init__(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

    @property
    def input_dir(self) -> Path:
        return self.input_data_dir / self.input_folder

    @property
    def output_dir(self) -> Path:
        return self.output_data_dir / self.output_folder
