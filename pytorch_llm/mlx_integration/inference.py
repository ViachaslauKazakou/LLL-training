"""
inference.py — генерация текста через MLX модели.

Ответственность: только инференс (отделён от trainer.py).
"""

import time
from pathlib import Path


def load_mlx_model(model_id: str, adapter_path: str | None = None):
    """
    Загружает MLX модель и токенизатор.

    Returns:
        (model, tokenizer)

    Raises:
        ImportError если mlx_lm не установлен
        Exception при ошибке загрузки
    """
    try:
        import mlx_lm
    except ImportError:
        raise ImportError(
            "mlx-lm не установлен. Установите: poetry add mlx-lm"
        )

    # Проверяем адаптер
    adapter = None
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if adapter_dir.exists() and any(adapter_dir.glob("*.safetensors")):
            adapter = adapter_path

    model, tokenizer = mlx_lm.load(model_id, adapter_path=adapter)
    return model, tokenizer


def mlx_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    system_prompt: str = "",
) -> tuple[str, dict]:
    """
    Генерирует текст через MLX модель.

    Интерфейс совместим с generate_text() из pytorch_llm/inference.py.

    Returns:
        (generated_text, stats) где stats содержит:
            - total_time: время генерации (сек)
            - tokens_generated: количество сгенерированных токенов
            - tokens_per_second: скорость
            - prompt_tokens: токены промпта
    """
    try:
        import mlx_lm
    except ImportError:
        raise ImportError("mlx-lm не установлен")

    # Формируем финальный промпт
    if system_prompt:
        # Пробуем применить chat template если доступен
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        formatted_prompt = prompt

    start_time = time.time()

    # Подсчёт токенов промпта
    prompt_tokens_ids = tokenizer.encode(formatted_prompt)
    prompt_tokens = len(prompt_tokens_ids)

    # Генерация
    generated_text = mlx_lm.generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        verbose=False,
    )

    total_time = time.time() - start_time

    # Убираем промпт из результата если он включён в ответ
    if generated_text.startswith(formatted_prompt):
        generated_text = generated_text[len(formatted_prompt):]

    tokens_generated = len(tokenizer.encode(generated_text))
    tokens_per_second = tokens_generated / total_time if total_time > 0 else 0

    stats = {
        "total_time": total_time,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
        "prompt_tokens": prompt_tokens,
    }

    return generated_text, stats
