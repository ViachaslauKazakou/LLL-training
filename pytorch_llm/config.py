"""
config.py — конфигурация модели и обучения

Все гиперпараметры в одном месте.
"""

import torch
from dataclasses import dataclass
import time


def get_device() -> str:
    """
    Автоматически выбирает лучшее устройство.
    
    Приоритет: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_mps_memory_stats() -> dict:
    """
    Получить статистику памяти MPS.
    
    Returns:
        dict с ключами: allocated_gb, reserved_gb, total_gb, usage_percent
        Или None, если MPS недоступен
    """
    if not torch.backends.mps.is_available():
        return None
    
    try:
        # MPS memory stats (в байтах)
        allocated = torch.mps.current_allocated_memory()
        reserved = torch.mps.driver_allocated_memory()
        
        # Конвертируем в GB
        allocated_gb = allocated / (1024**3)
        reserved_gb = reserved / (1024**3)
        
        # Примерная общая память (для M1/M2/M3 это unified memory)
        # Пробуем получить через psutil, если доступен
        try:
            import psutil
            total_memory = psutil.virtual_memory().total
        except ImportError:
            # Если psutil не установлен, используем примерную оценку
            # Для Apple Silicon обычно 8/16/32/64 GB
            total_memory = 16 * (1024**3)  # Предполагаем 16 GB по умолчанию
        
        total_gb = total_memory / (1024**3)
        usage_percent = (reserved / total_memory * 100) if total_memory > 0 else 0
        
        return {
            'allocated_gb': allocated_gb,
            'reserved_gb': reserved_gb,
            'total_gb': total_gb,
            'usage_percent': usage_percent
        }
    except Exception as e:
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'total_gb': 0.0,
            'usage_percent': 0.0,
            'error': str(e)
        }


def get_cuda_memory_stats() -> dict:
    """
    Получить статистику памяти CUDA.
    
    Returns:
        dict с ключами: allocated_gb, reserved_gb, total_gb, usage_percent
        Или None, если CUDA недоступен
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        total = torch.cuda.get_device_properties(0).total_memory
        
        allocated_gb = allocated / (1024**3)
        reserved_gb = reserved / (1024**3)
        total_gb = total / (1024**3)
        usage_percent = (reserved / total * 100) if total > 0 else 0
        
        return {
            'allocated_gb': allocated_gb,
            'reserved_gb': reserved_gb,
            'total_gb': total_gb,
            'usage_percent': usage_percent
        }
    except Exception as e:
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'total_gb': 0.0,
            'usage_percent': 0.0,
            'error': str(e)
        }


def get_memory_stats(device: str) -> dict:
    """
    Универсальная функция для получения статистики памяти.
    
    Args:
        device: "mps", "cuda" или "cpu"
    
    Returns:
        dict с памятью или None для CPU
    """
    if device == "mps":
        return get_mps_memory_stats()
    elif device == "cuda":
        return get_cuda_memory_stats()
    else:
        return None


def benchmark_device(device: str, size: int = 1024) -> dict:
    """
    Быстрый бенчмарк производительности device.
    
    Выполняет простую операцию матричного умножения и измеряет время.
    
    Args:
        device: "mps", "cuda" или "cpu"
        size: размер матрицы для теста
    
    Returns:
        dict с результатами: time_ms, gflops, устройство работает
    """
    try:
        # Создаем случайные матрицы
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warm-up (первый запуск может быть медленнее)
        _ = torch.matmul(a, b)
        
        # Синхронизация для точного измерения
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        
        # Бенчмарк
        n_iterations = 10
        start = time.time()
        
        for _ in range(n_iterations):
            c = torch.matmul(a, b)
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / n_iterations
        time_ms = elapsed * 1000
        
        # GFLOPS = (2 * N^3) / time / 10^9
        # Матричное умножение N×N требует 2*N^3 операций
        gflops = (2 * size ** 3) / elapsed / 1e9
        
        return {
            'device': device,
            'time_ms': time_ms,
            'gflops': gflops,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        return {
            'device': device,
            'time_ms': 0,
            'gflops': 0,
            'success': False,
            'error': str(e)
        }


@dataclass
class ModelConfig:
    """Параметры архитектуры трансформера."""
    
    # Размеры
    vocab_size: int = 10000      # размер словаря
    d_model: int = 256           # размерность эмбеддингов
    n_layers: int = 6            # количество слоёв трансформера
    n_heads: int = 8             # количество голов attention
    d_ff: int = 1024             # размер feed-forward слоя (обычно 4×d_model)
    context_len: int = 512       # максимальная длина контекста
    
    # Regularization
    dropout: float = 0.1         # dropout rate
    
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0       # gradient clipping
    
    # Другое
    pad_token_id: int = 0
    
    def __post_init__(self):
        """Валидация параметров."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) должен делиться на n_heads ({self.n_heads})"


@dataclass  
class TrainingConfig:
    """Параметры обучения."""
    
    n_epochs: int = 10
    eval_every: int = 200        # evaluate каждые N шагов (было 500, уменьшили для контроля)
    save_every: int = 500        # сохранять каждые N шагов
    log_every: int = 50          # логировать каждые N шагов
    patience: int = 10           # early stopping: остановка после N проверок без улучшения
    
    # Paths
    data_path: str = "data/corpus.txt"
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "auto"         # "auto", "mps", "cuda" или "cpu"
    
    # Data
    train_split: float = 0.9     # 90% train, 10% val


# Предустановленные конфигурации
def get_small_config() -> ModelConfig:
    """Маленькая модель (~13M параметров) — быстрое обучение."""
    return ModelConfig(
        vocab_size=10000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        context_len=256,
    )


def get_medium_config() -> ModelConfig:
    """Средняя модель (~50M параметров)."""
    return ModelConfig(
        vocab_size=10000,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        context_len=512,
    )


def get_base_config() -> ModelConfig:
    """Базовая модель (~117M параметров, как GPT-2 small)."""
    return ModelConfig(
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        context_len=1024,
    )
