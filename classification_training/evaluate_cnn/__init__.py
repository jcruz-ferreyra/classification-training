# train_cnn/__init__.py
from .model_evaluation import evaluate_cnn
from .types import CNNEvaluationContext

__all__ = [
    # coco_fetching
    "evaluate_cnn",
    # types
    "CNNEvaluationContext",
]