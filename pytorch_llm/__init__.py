"""
PyTorch LLM — полный трансформер с нуля

Использование:
    from pytorch_llm import GPTModel, Trainer, get_small_config
"""

from .config import ModelConfig, TrainingConfig, get_small_config, get_medium_config, get_base_config
from .model import GPTModel
from .attention import MultiHeadAttention, create_causal_mask
from .training import Trainer, continue_training
from .data import CharTokenizer, TextDataset, load_data

__all__ = [
    # Config
    'ModelConfig',
    'TrainingConfig',
    'get_small_config',
    'get_medium_config',
    'get_base_config',
    
    # Model
    'GPTModel',
    'MultiHeadAttention',
    'create_causal_mask',
    
    # Training
    'Trainer',
    'continue_training',
    
    # Data
    'CharTokenizer',
    'TextDataset',
    'load_data',
]

__version__ = '1.0.0'
