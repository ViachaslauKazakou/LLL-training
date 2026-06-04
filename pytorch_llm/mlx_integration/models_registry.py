"""
models_registry.py — реестр предобученных моделей для MLX fine-tuning.

Ответственность: только метаданные моделей.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelInfo:
    name: str           # Отображаемое имя
    hf_id: str          # HuggingFace ID (mlx-community/...)
    size_gb: float      # Приблизительный размер на диске (GB)
    params_label: str   # Метка параметров (1B, 3B, ...)
    description: str    # Краткое описание
    context_length: int # Максимальная длина контекста


CURATED_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="Qwen2.5 0.5B Instruct",
        hf_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        size_gb=0.4,
        params_label="0.5B",
        description="Самая маленькая модель — быстрый fine-tuning, подходит для ограниченной памяти",
        context_length=32768,
    ),
    ModelInfo(
        name="Llama 3.2 1B Instruct",
        hf_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
        size_gb=0.7,
        params_label="1B",
        description="Meta Llama 3.2 1B — хорошее качество при малом размере, ~700 MB на диске",
        context_length=131072,
    ),
    ModelInfo(
        name="Qwen2.5 1.5B Instruct",
        hf_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        size_gb=1.0,
        params_label="1.5B",
        description="Alibaba Qwen2.5 1.5B — отличный баланс качество/размер, поддерживает русский",
        context_length=32768,
    ),
    ModelInfo(
        name="SmolLM2 1.7B Instruct",
        hf_id="mlx-community/SmolLM2-1.7B-Instruct-4bit",
        size_gb=1.0,
        params_label="1.7B",
        description="HuggingFace SmolLM2 — компактная модель с хорошим следованием инструкциям",
        context_length=8192,
    ),
    ModelInfo(
        name="Llama 3.2 3B Instruct",
        hf_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
        size_gb=2.0,
        params_label="3B",
        description="Meta Llama 3.2 3B — заметно лучше 1B при приемлемом размере",
        context_length=131072,
    ),
    ModelInfo(
        name="Qwen2.5 3B Instruct",
        hf_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
        size_gb=2.0,
        params_label="3B",
        description="Alibaba Qwen2.5 3B — сильная multilingual модель, хорошо для русского текста",
        context_length=32768,
    ),
    ModelInfo(
        name="Gemma 2 2B Instruct",
        hf_id="mlx-community/gemma-2-2b-it-4bit",
        size_gb=1.4,
        params_label="2B",
        description="Google Gemma 2 2B — современная архитектура с хорошим качеством",
        context_length=8192,
    ),
    ModelInfo(
        name="Phi-3.5 Mini 3.8B Instruct",
        hf_id="mlx-community/Phi-3.5-mini-instruct-4bit",
        size_gb=2.3,
        params_label="3.8B",
        description="Microsoft Phi-3.5 — сильная модель от Microsoft, хорошее качество рассуждений",
        context_length=131072,
    ),
]


def is_model_cached(hf_id: str) -> bool:
    """Проверяет, скачана ли модель в кеш HuggingFace."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return False
    # HF хранит модели в формате models--owner--repo
    model_dir_name = "models--" + hf_id.replace("/", "--")
    model_path = cache_dir / model_dir_name
    return model_path.exists()


def get_model_display_label(m: ModelInfo) -> str:
    """Форматирует строку для selectbox с индикатором кеша."""
    cached = is_model_cached(m.hf_id)
    cached_indicator = "✅" if cached else "⬇️"
    return f"{cached_indicator} {m.name} ({m.params_label}, ~{m.size_gb} GB)"
