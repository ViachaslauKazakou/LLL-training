"""
inference.py — генерация текста через MLX модели.

Ответственность: только инференс (отделён от trainer.py).
"""

import time
from pathlib import Path


def _has_chat_template(tokenizer) -> bool:
    """Проверяет, есть ли у токенизатора нетривиальный chat template."""
    raw = getattr(tokenizer, "chat_template", None)
    if raw is None:
        # TokenizerWrapper хранит внутри tokenizer
        inner = getattr(tokenizer, "tokenizer", None)
        raw = getattr(inner, "chat_template", None)
    return bool(raw)


def _apply_chat_template(tokenizer, messages: list[dict]) -> str | None:
    """Применяет chat template. Возвращает None если шаблона нет."""
    if not _has_chat_template(tokenizer):
        return None
    try:
        result = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # на случай если вернулись байты
        if isinstance(result, bytes):
            result = result.decode("utf-8")
        return result
    except Exception:
        return None


def load_mlx_model(model_id: str, adapter_path: str | None = None):
    """
    Загружает MLX модель и токенизатор.

    Returns:
        (model, tokenizer)
    """
    try:
        import mlx_lm
    except ImportError:
        raise ImportError("mlx-lm не установлен. Установите: poetry add mlx-lm")

    adapter = None
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if adapter_dir.exists() and any(adapter_dir.glob("*.safetensors")):
            adapter = adapter_path

    model, tokenizer = mlx_lm.load(model_id.strip(), adapter_path=adapter)
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
    use_chat_template: bool = True,
) -> tuple[str, dict]:
    """
    Генерирует текст через MLX модель.

    Returns:
        (generated_text, stats)
    """
    try:
        import mlx_lm
    except ImportError:
        raise ImportError("mlx-lm не установлен")

    # Формируем промпт
    formatted_prompt = prompt
    used_template = False

    if use_chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = _apply_chat_template(tokenizer, messages)
        if result is not None:
            formatted_prompt = result
            used_template = True

    # Если chat template не применился, но есть system_prompt — добавляем вручную
    if not used_template and system_prompt:
        formatted_prompt = f"{system_prompt}\n\n{prompt}"

    start_time = time.time()

    # Подсчёт токенов промпта
    try:
        prompt_ids = tokenizer.encode(formatted_prompt)
        prompt_tokens = len(prompt_ids) if prompt_ids is not None else 0
    except Exception:
        prompt_tokens = 0

    from mlx_lm.sample_utils import make_sampler, make_logits_processors

    # repetition_penalty=1.0 — отключаем процессор (нет эффекта от него)
    rep_penalty = repetition_penalty if repetition_penalty != 1.0 else None

    generated_text = mlx_lm.generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature, top_p=top_p),
        logits_processors=make_logits_processors(repetition_penalty=rep_penalty),
        verbose=False,
    )

    total_time = time.time() - start_time

    # Некоторые версии mlx_lm включают промпт в результат
    if generated_text.startswith(formatted_prompt):
        generated_text = generated_text[len(formatted_prompt):]

    try:
        tokens_generated = len(tokenizer.encode(generated_text)) if generated_text else 0
    except Exception:
        tokens_generated = 0

    tokens_per_second = tokens_generated / total_time if total_time > 0 else 0

    stats = {
        "total_time": total_time,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
        "prompt_tokens": prompt_tokens,
        "used_chat_template": used_template,
    }

    return generated_text, stats
