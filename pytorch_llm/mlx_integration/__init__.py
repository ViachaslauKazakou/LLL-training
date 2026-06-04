"""
mlx_integration — изолированный модуль для fine-tuning через MLX на Apple Silicon.

Использование:
    from mlx_integration import MLX_AVAILABLE, CURATED_MODELS
    from mlx_integration import prepare_mlx_dataset, DataPrepConfig
    from mlx_integration import run_mlx_training, MLXTrainingConfig
    from mlx_integration import load_mlx_model, mlx_generate
"""

try:
    import mlx_lm  # noqa: F401
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from .models_registry import CURATED_MODELS, ModelInfo, is_model_cached, get_model_display_label
from .data_prep import DataPrepConfig, prepare_mlx_dataset
from .trainer import MLXTrainingConfig, run_mlx_training, fuse_adapter
from .inference import load_mlx_model, mlx_generate

__all__ = [
    "MLX_AVAILABLE",
    "CURATED_MODELS",
    "ModelInfo",
    "is_model_cached",
    "get_model_display_label",
    "DataPrepConfig",
    "prepare_mlx_dataset",
    "MLXTrainingConfig",
    "run_mlx_training",
    "fuse_adapter",
    "load_mlx_model",
    "mlx_generate",
]
