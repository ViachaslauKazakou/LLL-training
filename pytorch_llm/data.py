"""
data.py — загрузка и подготовка данных

Токенизация на уровне символов (char-level) для простоты.
Для production используйте BPE/WordPiece (tiktoken, sentencepiece).

Поддерживаемые форматы:
- .txt — обычный текстовый файл
- .json — форум/чат данные (структура: {"User": ..., "messages": {...}})
"""

import math
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Union
from logger import data_logger, log_session_start, log_session_end

# Импортируем токенизаторы из tokenizer.py
try:
    from tokenizer import (
        CharTokenizer as NewCharTokenizer,
        TikTokenizer,
        HybridChemTokenizer,
        create_tokenizer,
        normalize_chemistry_text,
    )
    HAS_NEW_TOKENIZERS = True
except ImportError:
    HAS_NEW_TOKENIZERS = False
    NewCharTokenizer = None
    TikTokenizer = None
    HybridChemTokenizer = None


class TextDataset(Dataset):
    """
    Dataset для обучения языковой модели.
    
    Данные: просто текстовый файл, разбитый на куски длины context_len.
    stride: шаг между окнами (по умолчанию = context_len, non-overlapping)
    """
    
    def __init__(self, text: str, context_len: int, tokenizer, stride: int = None):
        self.context_len = context_len
        self.tokenizer = tokenizer
        # Если stride не указан, используем non-overlapping windows
        self.stride = stride if stride is not None else context_len
        
        # Токенизируем весь текст
        if hasattr(tokenizer, 'encode'):
            # Новые токенизаторы (TikTokenizer, NewCharTokenizer)
            tokens = tokenizer.encode(text)
        else:
            # Legacy CharTokenizer (vocab dict)
            vocab = tokenizer.vocab
            tokens = [vocab.get(c, vocab['<unk>']) for c in text]
        
        self.data = torch.tensor(tokens, dtype=torch.long)
        
        # Проверка размера датасета
        if len(self.data) <= context_len:
            data_logger.warning(f"⚠️ Датасет слишком мал! Токенов: {len(self.data)}, нужно минимум: {context_len + 1}")
        
        # Рассчитываем количество примеров
        num_examples = max(0, (len(self.data) - self.context_len) // self.stride)
        
        data_logger.info(f"Dataset: {len(self.data):,} токенов → {num_examples:,} примеров (stride={self.stride})")
    
    def __len__(self) -> int:
        # Количество окон с заданным stride
        return max(0, (len(self.data) - self.context_len) // self.stride)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: (context_len,) — входная последовательность
            targets: (context_len,) — target последовательность (сдвинутая на 1)
        """
        start_idx = idx * self.stride
        chunk = self.data[start_idx : start_idx + self.context_len + 1]
        input_ids = chunk[:-1]
        targets = chunk[1:]
        return input_ids, targets


def auto_select_stride(
    token_count: int,
    context_len: int,
    target_windows: int = 1024
) -> int:
    """
    Выбирает stride автоматически.

    Для маленьких корпусов используем overlap, чтобы получить больше
    обучающих окон и не заканчивать эпоху за 2-3 batch'а.
    Для больших корпусов сохраняем stride=context_len.
    """
    available_tokens = token_count - context_len
    if available_tokens <= 0:
        return context_len

    # Если non-overlapping окна уже дают достаточно примеров, overlap не нужен.
    non_overlapping_windows = max(0, available_tokens // context_len)
    if non_overlapping_windows >= target_windows:
        return context_len

    max_stride_for_target = max(1, available_tokens // target_windows)
    return min(context_len, max_stride_for_target)


class CharTokenizer:
    """
    Простой char-level токенизатор.
    
    Для реальных задач используйте:
    - tiktoken (OpenAI)
    - sentencepiece (Google)
    - Hugging Face tokenizers
    """
    
    def __init__(self, text: str):
        # Собираем уникальные символы
        chars = sorted(set(text))
        
        # Специальные токены
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        
        # Создаём словарь
        self.vocab = {**{tok: i for i, tok in enumerate(self.special_tokens)},
                      **{c: i + len(self.special_tokens) for i, c in enumerate(chars)}}
        self.idx_to_char = {i: c for c, i in self.vocab.items()}
        
        data_logger.info(f"Токенизатор: {len(self.vocab)} токенов")
    
    def encode(self, text: str) -> list[int]:
        """Текст → список индексов."""
        return [self.vocab.get(c, self.vocab['<unk>']) for c in text]
    
    def decode(self, ids: list[int]) -> str:
        """Список индексов → текст."""
        return ''.join(self.idx_to_char.get(i, '<unk>') for i in ids)
    
    def vocab_size(self) -> int:
        """Размер словаря токенов."""
        return len(self.vocab)
    
    def to_dict(self) -> dict:
        """Сериализация для сохранения в checkpoint (для обратной совместимости)."""
        # Конвертируем vocab обратно в список символов
        vocab_list = ['<pad>', '<unk>', '<bos>', '<eos>']
        for char, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
            if char not in self.special_tokens:
                vocab_list.append(char)
        
        return {
            'type': 'char',
            'vocab': vocab_list
        }


def convert_forum_json_to_text(json_path: str, include_topics: bool = True) -> str:
    """
    Конвертирует JSON файл с форумными сообщениями в текст.
    
    Ожидаемая структура JSON:
    {
        "User": "123456",
        "messages": {
            "Тема 1": ["сообщение 1", "сообщение 2", ...],
            "Тема 2": ["сообщение 1", ...],
            ...
        }
    }
    
    Args:
        json_path: Путь к JSON файлу
        include_topics: Включать ли названия тем в текст (для контекста)
    
    Returns:
        Текстовая строка со всеми сообщениями
    """
    data_logger.info(f"Конвертация форумного JSON: {json_path}")
    data_logger.info(f"Параметр include_topics: {include_topics}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = data.get('messages', {})
    user_id = data.get('User', 'unknown')
    
    text_parts = []
    
    data_logger.info(f"Загрузка JSON: пользователь {user_id}")
    data_logger.info(f"JSON пользователь: {user_id}")
    data_logger.info(f"Найдено топиков: {len(messages)}")
    data_logger.info(f"Найдено топиков: {len(messages)}")
    
    total_msgs = 0
    for topic, msgs in messages.items():
        total_msgs += len(msgs)
        
        if include_topics:
            # Добавляем название темы как контекст
            text_parts.append(f"\n\n### {topic}\n\n")
        
        # Добавляем сообщения (каждое с новой строки)
        for msg in msgs:
            text_parts.append(msg.strip())
            text_parts.append('\n\n')
    
    data_logger.info(f"Всего сообщений: {total_msgs}")
    data_logger.info(f"Всего сообщений: {total_msgs}")
    
    result = ''.join(text_parts)
    data_logger.info(f"Извлечено символов: {len(result):,}")
    data_logger.info(f"Извлечено символов: {len(result):,}")
    
    return result


def convert_tasks_json_to_text(json_path: str) -> str:
    """
    Конвертирует JSON файл с задачами (из OCR) в текст.
    
    Ожидаемая структура JSON:
    {
        "dataset_name": "chemistry_tasks_...",
        "tasks": [
            {
                "question": "...",
                "solution": "...",
                "answer": "...",
                "category": "...",
                ...
            },
            ...
        ]
    }
    
    Args:
        json_path: Путь к JSON файлу
    
    Returns:
        Текстовая строка со всеми задачами
    """
    data_logger.info(f"Конвертация задач JSON: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tasks = data.get('tasks', [])
    dataset_name = data.get('dataset_name', 'unknown')
    
    text_parts = []
    
    data_logger.info(f"Загрузка JSON: датасет {dataset_name}")
    data_logger.info(f"JSON датасет: {dataset_name}")
    data_logger.info(f"Найдено задач: {len(tasks)}")
    data_logger.info(f"Найдено задач: {len(tasks)}")
    
    for i, task in enumerate(tasks, 1):
        question = task.get('question', '').strip()
        solution = task.get('solution', '').strip()
        answer = task.get('answer', '').strip()
        category = task.get('category', '')
        
        # Формируем текст задачи в формате диалога
        if category and category != 'другое':
            text_parts.append(f"\n\n### Задача {i} ({category})\n\n")
        else:
            text_parts.append(f"\n\n### Задача {i}\n\n")
        
        # Вопрос
        if question:
            text_parts.append(f"Вопрос: {question}\n\n")
        
        # Решение
        if solution:
            text_parts.append(f"Решение:\n{solution}\n\n")
        
        # Ответ
        if answer:
            text_parts.append(f"Ответ: {answer}\n\n")
    
    result = ''.join(text_parts)
    data_logger.info(f"Извлечено символов: {len(result):,}")
    data_logger.info(f"Извлечено символов: {len(result):,}")
    
    if len(result) == 0:
        data_logger.warning("⚠️ Пустой результат конвертации задач!")
        data_logger.warning("⚠️ Внимание: задачи пустые или некорректный формат JSON")
    
    return result


def convert_interview_json_to_text(json_path: str) -> str:
    """
    Конвертирует JSON файл с вопросами для интервью в текст.
    
    Ожидаемая структура JSON:
    {
        "items": [
            {
                "topic": "последовательности",
                "source_type": "import_api",
                "content": "Вопрос: ...\n\nЭталонный ответ:\n..."
            },
            ...
        ]
    }
    
    Args:
        json_path: Путь к JSON файлу
    
    Returns:
        Текстовая строка со всеми вопросами и ответами
    """
    data_logger.info(f"Конвертация интервью JSON: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get('items', [])
    
    text_parts = []
    
    data_logger.info(f"Загрузка JSON: вопросы для интервью")
    data_logger.info(f"JSON формат: интервью")
    data_logger.info(f"Найдено вопросов: {len(items)}")
    data_logger.info(f"Найдено вопросов: {len(items)}")
    
    current_topic = None
    
    for i, item in enumerate(items, 1):
        topic = item.get('topic', '').strip()
        content = item.get('content', '').strip()
        
        if not content:
            continue
        
        # Добавляем заголовок темы при смене
        if topic and topic != current_topic:
            text_parts.append(f"\n\n## Тема: {topic}\n\n")
            current_topic = topic
        
        # Добавляем контент вопроса
        text_parts.append(f"{content}\n\n")
        text_parts.append("---\n\n")
    
    result = ''.join(text_parts)
    data_logger.info(f"Извлечено символов: {len(result):,}")
    data_logger.info(f"Извлечено символов: {len(result):,}")
    
    if len(result) == 0:
        data_logger.warning("⚠️ Пустой результат конвертации интервью!")
        data_logger.warning("⚠️ Внимание: вопросы пустые или некорректный формат JSON")
    
    return result


def detect_file_format(data_path: str) -> str:
    """
    Определяет формат файла по расширению.
    
    Returns:
        'json' или 'text'
    """
    path = Path(data_path)
    suffix = path.suffix.lower()
    
    if suffix == '.json':
        return 'json'
    else:
        return 'text'


def detect_json_type(json_path: str) -> str:
    """
    Определяет тип JSON (форумный, задачи или интервью).
    
    Returns:
        'forum' — форумные сообщения ({"User": ..., "messages": {...}})
        'tasks' — задачи из OCR ({"tasks": [...]})
        'interview' — вопросы для интервью ({"items": [...]})
        'unknown' — неизвестный формат
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Проверяем наличие ключей
        if 'tasks' in data:
            return 'tasks'
        elif 'items' in data:
            return 'interview'
        elif 'messages' in data or 'User' in data:
            return 'forum'
        else:
            return 'unknown'
    except Exception as e:
        data_logger.error(f"Ошибка определения типа JSON: {e}")
        return 'unknown'


def load_data(
    data_path: str,
    context_len: int,
    batch_size: int,
    train_split: float = 0.9,
    include_topics: bool = True,
    stride: int | None = None,
    tokenizer_type: str = 'char',
    tokenizer_encoding: str = 'cl100k_base',
    tokenizer_config: dict | None = None,
    normalize_chemistry: bool = True
) -> tuple[DataLoader, DataLoader, object]:
    """
    Загружает данные и создаёт DataLoader'ы.
    
    Поддерживаемые форматы:
    - .txt — обычный текст
    - .json — автоматически определяет тип:
        * 'tasks' — задачи из OCR ({"tasks": [...]})
        * 'interview' — вопросы для интервью ({"items": [...]})
        * 'forum' — форумные сообщения ({"User": ..., "messages": {...}})
    
    Args:
        data_path: Путь к файлу с данными
        context_len: Длина контекста
        batch_size: Размер батча
        train_split: Доля для train (остальное — validation)
        include_topics: Для форумного JSON — включать ли названия топиков
        stride: Шаг между окнами. Если не задан, выбирается автоматически.
        tokenizer_type: Тип токенизатора ('char', 'tiktoken' или 'hybrid')
        tokenizer_encoding: Для tiktoken - имя энкодера (cl100k_base, o200k_base, p50k_base)
        tokenizer_config: Конфиг токенизатора из checkpoint (для fine-tuning без пересборки vocab)
        normalize_chemistry: Нормализовать химическую запись до токенизации
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    log_session_start(data_logger, "Загрузка данных")
    data_logger.info(f"Путь к данным: {data_path}")
    data_logger.info(f"Параметры: context_len={context_len}, batch_size={batch_size}, train_split={train_split}")
    
    # Определяем формат файла
    file_format = detect_file_format(data_path)
    data_logger.info(f"Формат данных: {file_format}")
    data_logger.info(f"Формат данных: {file_format}")
    
    # Читаем текст
    if file_format == 'json':
        # Определяем тип JSON
        json_type = detect_json_type(data_path)
        data_logger.info(f"Тип JSON: {json_type}")
        data_logger.info(f"Тип JSON: {json_type}")
        
        if json_type == 'tasks':
            # Формат задач (из OCR приложения)
            text = convert_tasks_json_to_text(data_path)
        elif json_type == 'interview':
            # Формат вопросов для интервью
            text = convert_interview_json_to_text(data_path)
        elif json_type == 'forum':
            # Форумный формат
            text = convert_forum_json_to_text(data_path, include_topics=include_topics)
        else:
            raise ValueError(f"Неизвестный тип JSON. Ожидается 'tasks', 'interview' или 'forum'. Проверьте структуру файла {data_path}")
    else:
        text = Path(data_path).read_text(encoding='utf-8')
        data_logger.info(f"Загружен текстовый файл: {len(text):,} символов")
    
    data_logger.info(f"Общий размер текста: {len(text):,} символов")
    data_logger.info(f"Общий размер текста: {len(text):,} символов")
    
    if len(text) == 0:
        raise ValueError("Пустой текст после загрузки! Проверьте содержимое файла.")

    if normalize_chemistry and HAS_NEW_TOKENIZERS:
        normalized_text = normalize_chemistry_text(text)
        if normalized_text != text:
            data_logger.info("🧪 Chemistry normalization применена перед токенизацией")
            data_logger.info(f"  Символов до/после: {len(text):,} -> {len(normalized_text):,}")
        text = normalized_text
    
    # Создаём токенизатор
    if tokenizer_config and isinstance(tokenizer_config, dict):
        cfg_type = tokenizer_config.get('type')

        if cfg_type == 'tiktoken' and HAS_NEW_TOKENIZERS:
            tokenizer = TikTokenizer.from_dict(tokenizer_config)
            data_logger.info(
                f"TikTokenizer загружен из checkpoint: {tokenizer.vocab_size()} токенов "
                f"({tokenizer_config.get('encoding_name', tokenizer_encoding)})"
            )
            data_logger.info(f"Токенизатор: {tokenizer.vocab_size()} токенов (TikToken checkpoint)")
        elif cfg_type == 'hybrid' and HAS_NEW_TOKENIZERS:
            tokenizer = HybridChemTokenizer.from_dict(tokenizer_config)
            data_logger.info(f"HybridTokenizer загружен из checkpoint: {tokenizer.vocab_size()} токенов")
            data_logger.info(f"Токенизатор: {tokenizer.vocab_size()} токенов (Hybrid checkpoint)")
        elif cfg_type == 'char' and HAS_NEW_TOKENIZERS:
            tokenizer = NewCharTokenizer.from_dict(tokenizer_config)
            data_logger.info(f"CharTokenizer загружен из checkpoint: {tokenizer.vocab_size()} токенов (new)")
            data_logger.info(f"Токенизатор: {tokenizer.vocab_size()} токенов (Char checkpoint)")
        elif cfg_type == 'char' and 'vocab' in tokenizer_config:
            # Legacy CharTokenizer: фиксируем vocab из checkpoint без пересборки по новому тексту.
            vocab_chars = [c for c in tokenizer_config['vocab'] if c not in {'<pad>', '<unk>', '<bos>', '<eos>'}]
            tokenizer = CharTokenizer(''.join(vocab_chars))
            data_logger.info(f"CharTokenizer загружен из checkpoint: {len(tokenizer.vocab)} токенов (legacy)")
            data_logger.info(f"Токенизатор: {len(tokenizer.vocab)} токенов (Char legacy checkpoint)")
        else:
            data_logger.warning("⚠️ tokenizer_config из checkpoint не распознан, используем создание токенизатора по данным")
            tokenizer_config = None

    if not tokenizer_config:
        if HAS_NEW_TOKENIZERS and tokenizer_type in ['tiktoken', 'char_new', 'hybrid']:
            # Используем новые токенизаторы из tokenizer.py
            if tokenizer_type == 'tiktoken':
                tokenizer = TikTokenizer(encoding_name=tokenizer_encoding)
                data_logger.info(f"TikTokenizer создан: {tokenizer.vocab_size()} токенов ({tokenizer_encoding})")
                data_logger.info(f"Токенизатор: {tokenizer.vocab_size()} токенов (TikToken {tokenizer_encoding})")
            elif tokenizer_type == 'hybrid':
                tokenizer = HybridChemTokenizer.from_text(text)
                data_logger.info(f"HybridTokenizer создан: {tokenizer.vocab_size()} токенов")
                data_logger.info(f"Токенизатор: {tokenizer.vocab_size()} токенов (Hybrid)")
            else:  # char_new
                tokenizer = NewCharTokenizer.from_text(text)
                data_logger.info(f"CharTokenizer создан: {tokenizer.vocab_size()} токенов (new)")
                data_logger.info(f"Токенизатор: {tokenizer.vocab_size()} токенов (Char)")
        else:
            # Fallback: используем старый CharTokenizer для обратной совместимости
            tokenizer = CharTokenizer(text)
            data_logger.info(f"CharTokenizer создан (legacy): {len(tokenizer.vocab)} токенов")
            data_logger.info(f"Токенизатор: {len(tokenizer.vocab)} токенов (Char legacy)")

    # Метрики качества токенизации
    tokenized_full = tokenizer.encode(text)
    n_tokens = max(1, len(tokenized_full))
    chars_per_token = len(text) / n_tokens
    data_logger.info(f"📏 Tokenization quality: {n_tokens:,} токенов, {chars_per_token:.2f} char/token")

    if hasattr(tokenizer, 'unk_id'):
        unk_id = tokenizer.unk_id
        unk_count = sum(1 for tok in tokenized_full if tok == unk_id)
        unk_share = (unk_count / n_tokens) * 100
        data_logger.info(f"❓ Unknown token share: {unk_count:,}/{n_tokens:,} ({unk_share:.2f}%)")
    
    # Разбиваем на train/val
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    if stride is None:
        if hasattr(tokenizer, 'encode'):
            train_token_count = len(tokenizer.encode(train_text))
            val_token_count = len(tokenizer.encode(val_text))
        else:
            train_token_count = len(train_text)
            val_token_count = len(val_text)

        train_stride = auto_select_stride(train_token_count, context_len, target_windows=1024)
        val_stride = auto_select_stride(val_token_count, context_len, target_windows=256)
    else:
        train_stride = stride
        val_stride = stride

    estimated_train_windows = max(0, (max(train_token_count - context_len, 0)) // train_stride) if 'train_token_count' in locals() else None
    estimated_val_windows = max(0, (max(val_token_count - context_len, 0)) // val_stride) if 'val_token_count' in locals() else None
    data_logger.info(
        f"Stride: train={train_stride}, val={val_stride}"
        + (
            f" → ожидаемо окон train={estimated_train_windows}, val={estimated_val_windows}"
            if estimated_train_windows is not None and estimated_val_windows is not None else ""
        )
    )
    
    # Создаём датасеты
    train_dataset = TextDataset(train_text, context_len, tokenizer, stride=train_stride)
    val_dataset = TextDataset(val_text, context_len, tokenizer, stride=val_stride)
    data_logger.info(f"Датасеты созданы: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # Проверяем размер датасетов
    if len(train_dataset) == 0:
        recommended_context = max(16, len(train_text) // 2)
        raise ValueError(
            f"Train датасет пустой! Данных недостаточно.\n"
            f"Токенов в train: {len(train_text)}, требуется: {context_len + 1}\n"
            f"Решения:\n"
            f"1. Уменьшите context_len до {recommended_context} или меньше\n"
            f"2. Добавьте больше задач в JSON файл (сейчас текста: {len(text)} символов)"
        )
    
    if len(val_dataset) == 0:
        data_logger.warning("⚠️ Validation датасет пустой, будет использован train для валидации")
        data_logger.warning("⚠️ Validation датасет пуст — используем train для валидации")
        val_dataset = train_dataset  # Используем train как val
    
    data_logger.info(f"✅ Train: {len(train_dataset)} окон, Val: {len(val_dataset)} окон")

    if len(train_dataset) < batch_size * 10:
        recommended_batch_size = max(1, min(batch_size, len(train_dataset) // 10 or 1))
        data_logger.warning(
            "⚠️ Очень мало train batch'ей. Для более стабильного обучения "
            f"уменьшите batch_size примерно до {recommended_batch_size} или используйте больший корпус."
        )
    
    # pin_memory ускоряет передачу данных CPU → GPU, но:
    # - MPS (Apple Silicon) не поддерживает pin_memory
    # - CPU не использует pin_memory
    # Включаем только для CUDA (NVIDIA GPU)
    use_pin_memory = torch.cuda.is_available()
    if use_pin_memory:
        data_logger.info("pin_memory включен (CUDA)")
    else:
        data_logger.info("pin_memory отключен (MPS/CPU не поддерживают)")
    
    # Создаём DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # для простоты; можно увеличить
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    data_logger.info(f"DataLoader'ы созданы: train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    log_session_end(data_logger, "Загрузка данных")
    
    return train_loader, val_loader, tokenizer


def prepare_sample_data(output_path: str = "data/sample.txt"):
    """
    Создаёт маленький датасет для быстрого тестирования.
    
    В реальности используйте:
    - Wikipedia dump
    - Common Crawl
    - Книги (Project Gutenberg)
    - Свои тексты
    """
    sample_text = """
Искусственный интеллект — это область компьютерных наук, которая занимается созданием 
интеллектуальных машин, способных работать и реагировать как люди. Некоторые виды 
деятельности, для которых предназначены компьютеры с искусственным интеллектом, 
включают распознавание речи, обучение, планирование и решение проблем.

Машинное обучение — это метод анализа данных, который автоматизирует построение 
аналитических моделей. Это ветвь искусственного интеллекта, основанная на идее, 
что системы могут учиться на данных, выявлять закономерности и принимать решения 
с минимальным вмешательством человека.

Глубокое обучение — это подмножество машинного обучения в искусственном интеллекте, 
которое имеет сети, способные к обучению без присмотра на неразмеченных или 
неструктурированных данных. Также известно как глубокое нейронное обучение или 
глубокие нейронные сети.
""" * 10  # Повторяем 10 раз для большего объёма
    
    Path(output_path).parent.mkdir(exist_ok=True)
    Path(output_path).write_text(sample_text, encoding='utf-8')
    data_logger.info(f"Создан sample датасет: {output_path}")
