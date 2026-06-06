"""
data_prep.py — конвертация данных корпуса в JSONL для MLX fine-tuning.

Ответственность: только трансформация данных (txt/json → JSONL).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataPrepConfig:
    source_path: str
    output_dir: str
    dataset_mode: str = "auto"      # "auto" | "corpus" | "tasks"
    format: str = "completion"       # "completion" | "chat"
    train_split: float = 0.9
    auto_train_split: bool = True
    max_seq_length: int = 1024
    clean_forum: bool = False
    system_prompt: str = ""          # опционально для chat формата
    min_length: int = 20             # минимальная длина текста (символы)
    deduplicate: bool = True
    random_seed: int = 42


def _clean_forum_text(text: str) -> str:
    """Удаляет форумный шум: username#:, имена авторов, лишние пробелы."""
    # Убираем паттерны "username#123:" в начале строк
    text = re.sub(r'^[^\s:]{1,30}#\d+:\s*', '', text, flags=re.MULTILINE)
    # Убираем паттерны "Имя Фамилия:" в начале строк (2-3 слова + двоеточие)
    text = re.sub(r'^[A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+){0,2}:\s*', '', text, flags=re.MULTILINE)
    # Убираем многократные пустые строки
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """Консервативная оценка числа токенов.

    Используем коэффициент 3 вместо 4: русский текст + технические термины/формулы
    токенизируются плотнее. Запас ~25% защищает от варнинга 'longer than N tokens'.
    """
    return len(text) // 3


def _split_by_sentences(text: str, max_seq_length: int) -> list[str]:
    """Разбивает текст по предложениям когда абзац слишком длинный."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = _estimate_tokens(sent)

        # Одно предложение само по себе слишком длинное — режем по словам
        if sent_tokens > max_seq_length:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            words = sent.split()
            part: list[str] = []
            part_len = 0
            for word in words:
                wt = _estimate_tokens(word)
                if part_len + wt > max_seq_length and part:
                    chunks.append(" ".join(part))
                    part, part_len = [], 0
                part.append(word)
                part_len += wt
            if part:
                chunks.append(" ".join(part))
            continue

        if current_len + sent_tokens > max_seq_length and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0

        current.append(sent)
        current_len += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_text_into_chunks(text: str, max_seq_length: int) -> list[str]:
    """Разбивает текст на чанки по абзацам/предложениям с учётом max_seq_length."""
    paragraphs = re.split(r'\n\n+', text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = _estimate_tokens(para)

        # Абзац сам по себе слишком длинный — дробим по предложениям
        if para_tokens > max_seq_length:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            chunks.extend(_split_by_sentences(para, max_seq_length))
            continue

        if current_len + para_tokens > max_seq_length and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0

        current.append(para)
        current_len += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _extract_task_blocks(text: str) -> list[str]:
    """Извлекает цельные блоки задач вида 'Задача N: ...'."""
    starts = list(re.finditer(r'^Задача\s+\d+\s*:', text, re.MULTILINE))
    if not starts:
        return []

    blocks: list[str] = []
    for idx, match in enumerate(starts):
        start = match.start()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
        block = text[start:end].strip()
        block = re.sub(r'^(?:---\s*)+', '', block, flags=re.MULTILINE).strip()
        block = re.sub(r'(?:\s*---\s*)+$', '', block, flags=re.MULTILINE).strip()
        if block:
            blocks.append(block)
    return blocks


def _detect_dataset_mode(source_path: str, clean_forum: bool) -> str:
    """Автоматически определяет режим подготовки по содержимому источника."""
    path = Path(source_path)
    sample_text = ""

    if path.suffix.lower() == ".txt" and path.exists():
        sample_text = path.read_text(encoding="utf-8")
    else:
        raw_texts = _load_source_texts(source_path, clean_forum)
        sample_text = "\n\n".join(raw_texts[:200])

    if clean_forum:
        sample_text = _clean_forum_text(sample_text)

    task_blocks = _extract_task_blocks(sample_text)
    # Порог защищает от случайного единичного "Задача 1" в обычном тексте.
    return "tasks" if len(task_blocks) >= 5 else "corpus"


def _deduplicate_preserve_order(texts: list[str]) -> list[str]:
    """Удаляет дубли, сохраняя исходный порядок."""
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        key = text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _split_train_valid(items: list[str], train_split: float) -> tuple[list[str], list[str]]:
    """Разбивает на train/valid без пересечений."""
    count = len(items)
    if count == 1:
        return items, []

    valid_count = max(1, int(round(count * (1.0 - train_split))))
    valid_count = min(valid_count, count - 1)
    split_idx = count - valid_count
    return items[:split_idx], items[split_idx:]


def _recommend_train_split(total_items: int, dataset_mode: str) -> float:
    """Рекомендует долю train в зависимости от размера датасета и режима."""
    if total_items <= 1:
        return 1.0

    if dataset_mode == "tasks":
        if total_items < 50:
            return 0.8
        if total_items < 200:
            return 0.9
        return 0.95

    # Для corpus обычно примеров больше, поэтому оставляем больше данных на train.
    if total_items < 200:
        return 0.9
    if total_items < 1000:
        return 0.95
    return 0.98


def _load_source_texts(source_path: str, clean_forum: bool) -> list[str]:
    """Читает txt или json файл, возвращает список текстовых блоков."""
    path = Path(source_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {source_path}")

    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        texts = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Пробуем разные поля
                    text = item.get("text") or item.get("content") or item.get("message") or ""
                    topic = item.get("topic") or item.get("title") or ""
                    if topic:
                        text = f"{topic}\n\n{text}"
                    if text.strip():
                        texts.append(text.strip())
                elif isinstance(item, str):
                    texts.append(item)
        elif isinstance(data, dict):
            # Формат {topic: [messages]}
            for topic, messages in data.items():
                if isinstance(messages, list):
                    for msg in messages:
                        msg_text = msg if isinstance(msg, str) else msg.get("text", "")
                        if msg_text.strip():
                            texts.append(f"{topic}\n\n{msg_text.strip()}")
    else:
        # Обычный текстовый файл
        with open(path, encoding="utf-8") as f:
            content = f.read()

        if clean_forum:
            content = _clean_forum_text(content)

        # Разбиваем на абзацы
        texts = [p.strip() for p in re.split(r'\n\n+', content) if p.strip()]

    if clean_forum and path.suffix.lower() == ".json":
        texts = [_clean_forum_text(t) for t in texts]

    return texts


def _load_task_texts(source_path: str, clean_forum: bool) -> list[str]:
    """Загружает корпус и возвращает только цельные блоки задач."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {source_path}")

    if path.suffix.lower() == ".txt":
        content = path.read_text(encoding="utf-8")
        if clean_forum:
            content = _clean_forum_text(content)
        tasks = _extract_task_blocks(content)
    else:
        tasks = []
        raw_texts = _load_source_texts(source_path, clean_forum)
        for text in raw_texts:
            tasks.extend(_extract_task_blocks(text))

    if not tasks:
        raise ValueError(
            "Режим 'tasks' не нашел блоков вида 'Задача N: ...'. "
            "Используйте режим 'corpus' или проверьте формат исходного файла."
        )

    return tasks


def _make_completion_record(text: str) -> dict:
    """Создаёт запись в completion формате."""
    return {"text": text}


def _make_chat_record(text: str, system_prompt: str, dataset_mode: str) -> dict:
    """Создаёт запись в chat формате с system/user/assistant структурой."""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if dataset_mode == "tasks":
        parts = re.split(r'\n\s*Решение:\s*\n', text, maxsplit=1)
        if len(parts) == 2:
            user_part = parts[0].strip()
            assistant_part = parts[1].strip()
        else:
            user_part = text.strip()
            assistant_part = text.strip()
    else:
        # Для корпуса без явных реплик используем первое предложение как "вопрос",
        # остальное — как "ответ"
        sentences = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)
        if len(sentences) >= 2:
            user_part = sentences[0]
            assistant_part = sentences[1]
        else:
            # Нет явного разделения — кладём весь текст как assistant
            user_part = "Продолжи текст:"
            assistant_part = text

    messages.append({"role": "user", "content": user_part})
    messages.append({"role": "assistant", "content": assistant_part})

    return {"messages": messages}


def prepare_mlx_dataset(config: DataPrepConfig) -> dict:
    """
    Конвертирует корпус (txt/json) в train.jsonl + valid.jsonl для mlx_lm.

    Returns:
        dict с ключами: train_count, valid_count, output_dir, train_path, valid_path
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.dataset_mode not in {"auto", "corpus", "tasks"}:
        raise ValueError(f"Неподдерживаемый dataset_mode: {config.dataset_mode}")

    effective_mode = config.dataset_mode
    if effective_mode == "auto":
        effective_mode = _detect_dataset_mode(config.source_path, config.clean_forum)

    if effective_mode == "tasks":
        all_items = _load_task_texts(config.source_path, config.clean_forum)
    else:
        # Загружаем исходные тексты
        raw_texts = _load_source_texts(config.source_path, config.clean_forum)

        # Разбиваем длинные тексты на чанки
        all_items = []
        for text in raw_texts:
            chunks = _split_text_into_chunks(text, config.max_seq_length)
            all_items.extend(chunks)

        # Фильтруем по минимальной длине
        all_items = [c for c in all_items if len(c) >= config.min_length]

    if not all_items:
        raise ValueError(f"После обработки не осталось текстовых блоков. Проверьте файл: {config.source_path}")

    before_dedup = len(all_items)
    if config.deduplicate:
        all_items = _deduplicate_preserve_order(all_items)
    dedup_removed = before_dedup - len(all_items)

    # Перемешиваем детерминированно для воспроизводимого split
    import random
    rnd = random.Random(config.random_seed)
    rnd.shuffle(all_items)

    if config.auto_train_split:
        effective_train_split = _recommend_train_split(len(all_items), effective_mode)
    else:
        effective_train_split = float(config.train_split)

    effective_train_split = max(0.5, min(0.99, effective_train_split))

    # Разбиваем на train/valid без пересечений
    train_chunks, valid_chunks = _split_train_valid(all_items, effective_train_split)

    # Создаём записи в нужном формате
    def make_record(text: str) -> dict:
        if config.format == "chat":
            return _make_chat_record(text, config.system_prompt, effective_mode)
        else:
            return _make_completion_record(text)

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for chunk in train_chunks:
            f.write(json.dumps(make_record(chunk), ensure_ascii=False) + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for chunk in valid_chunks:
            f.write(json.dumps(make_record(chunk), ensure_ascii=False) + "\n")

    def _is_full_task(text: str) -> bool:
        normalized = text.strip()
        return normalized.startswith("Задача ") and "Решение:" in normalized and "Ответ:" in normalized

    full_task_train = sum(1 for item in train_chunks if _is_full_task(item))
    full_task_valid = sum(1 for item in valid_chunks if _is_full_task(item))

    return {
        "train_count": len(train_chunks),
        "valid_count": len(valid_chunks),
        "output_dir": str(output_dir),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "total_chunks": len(all_items),
        "dataset_mode": config.dataset_mode,
        "dataset_mode_effective": effective_mode,
        "train_split_effective": effective_train_split,
        "train_split_mode": "auto" if config.auto_train_split else "manual",
        "deduplicated_removed": dedup_removed,
        "full_task_train": full_task_train,
        "full_task_valid": full_task_valid,
    }
