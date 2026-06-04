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
    format: str = "completion"       # "completion" | "chat"
    train_split: float = 0.9
    max_seq_length: int = 1024
    clean_forum: bool = False
    system_prompt: str = ""          # опционально для chat формата
    min_length: int = 20             # минимальная длина текста (символы)


def _clean_forum_text(text: str) -> str:
    """Удаляет форумный шум: username#:, имена авторов, лишние пробелы."""
    # Убираем паттерны "username#123:" в начале строк
    text = re.sub(r'^[^\s:]{1,30}#\d+:\s*', '', text, flags=re.MULTILINE)
    # Убираем паттерны "Имя Фамилия:" в начале строк (2-3 слова + двоеточие)
    text = re.sub(r'^[A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+){0,2}:\s*', '', text, flags=re.MULTILINE)
    # Убираем многократные пустые строки
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _split_text_into_chunks(text: str, max_seq_length: int) -> list[str]:
    """Разбивает текст на чанки по абзацам/предложениям с учётом max_seq_length."""
    # Разбиваем по двойным переносам (абзацы)
    paragraphs = re.split(r'\n\n+', text)

    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Грубая оценка: 1 токен ≈ 3-4 символа
        para_tokens = len(para) // 3

        if current_len + para_tokens > max_seq_length and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= 20:
                chunks.append(chunk_text)
            current_chunk = [para]
            current_len = para_tokens
        else:
            current_chunk.append(para)
            current_len += para_tokens

    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        if len(chunk_text) >= 20:
            chunks.append(chunk_text)

    return chunks


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


def _make_completion_record(text: str) -> dict:
    """Создаёт запись в completion формате."""
    return {"text": text}


def _make_chat_record(text: str, system_prompt: str) -> dict:
    """Создаёт запись в chat формате с system/user/assistant структурой."""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

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

    # Загружаем исходные тексты
    raw_texts = _load_source_texts(config.source_path, config.clean_forum)

    # Разбиваем длинные тексты на чанки
    all_chunks = []
    for text in raw_texts:
        chunks = _split_text_into_chunks(text, config.max_seq_length)
        all_chunks.extend(chunks)

    # Фильтруем по минимальной длине
    all_chunks = [c for c in all_chunks if len(c) >= config.min_length]

    if not all_chunks:
        raise ValueError(f"После обработки не осталось текстовых блоков. Проверьте файл: {config.source_path}")

    # Перемешиваем для равномерного распределения
    import random
    random.shuffle(all_chunks)

    # Разбиваем на train/valid
    split_idx = max(1, int(len(all_chunks) * config.train_split))
    train_chunks = all_chunks[:split_idx]
    valid_chunks = all_chunks[split_idx:] if split_idx < len(all_chunks) else all_chunks[-max(1, len(all_chunks)//10):]

    # Создаём записи в нужном формате
    def make_record(text: str) -> dict:
        if config.format == "chat":
            return _make_chat_record(text, config.system_prompt)
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

    return {
        "train_count": len(train_chunks),
        "valid_count": len(valid_chunks),
        "output_dir": str(output_dir),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "total_chunks": len(all_chunks),
    }
