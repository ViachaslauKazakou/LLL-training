"""
data.py — загрузка и подготовка данных

Токенизация на уровне символов (char-level) для простоты.
Для production используйте BPE/WordPiece (tiktoken, sentencepiece).

Поддерживаемые форматы:
- .txt — обычный текстовый файл
- .json — форум/чат данные (структура: {"User": ..., "messages": {...}})
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from logger import data_logger, log_session_start, log_session_end


class TextDataset(Dataset):
    """
    Dataset для обучения языковой модели.
    
    Данные: просто текстовый файл, разбитый на куски длины context_len.
    """
    
    def __init__(self, text: str, context_len: int, vocab: dict[str, int]):
        self.context_len = context_len
        self.vocab = vocab
        
        # Токенизируем весь текст
        self.data = torch.tensor(
            [vocab.get(c, vocab['<unk>']) for c in text],
            dtype=torch.long
        )
        
        print(f"Dataset: {len(self.data):,} токенов")
        
        # Проверка размера датасета
        if len(self.data) <= context_len:
            data_logger.warning(f"⚠️ Датасет слишком мал! Токенов: {len(self.data)}, нужно минимум: {context_len + 1}")
            print(f"⚠️ Внимание: датасет слишком мал ({len(self.data)} токенов), нужно минимум {context_len + 1}")
    
    def __len__(self) -> int:
        # Количество окон длины context_len + 1 (target = след. токен)
        # Защита от отрицательных значений для маленьких датасетов
        return max(0, len(self.data) - self.context_len)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: (context_len,) — входная последовательность
            targets: (context_len,) — target последовательность (сдвинутая на 1)
        """
        chunk = self.data[idx : idx + self.context_len + 1]
        input_ids = chunk[:-1]
        targets = chunk[1:]
        return input_ids, targets


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
        
        print(f"Токенизатор: {len(self.vocab)} токенов")
    
    def encode(self, text: str) -> list[int]:
        """Текст → список индексов."""
        return [self.vocab.get(c, self.vocab['<unk>']) for c in text]
    
    def decode(self, ids: list[int]) -> str:
        """Список индексов → текст."""
        return ''.join(self.idx_to_char.get(i, '<unk>') for i in ids)


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
    
    print(f"Загрузка JSON: пользователь {user_id}")
    data_logger.info(f"JSON пользователь: {user_id}")
    print(f"Найдено топиков: {len(messages)}")
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
    
    print(f"Всего сообщений: {total_msgs}")
    data_logger.info(f"Всего сообщений: {total_msgs}")
    
    result = ''.join(text_parts)
    print(f"Извлечено символов: {len(result):,}")
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
    
    print(f"Загрузка JSON: датасет {dataset_name}")
    data_logger.info(f"JSON датасет: {dataset_name}")
    print(f"Найдено задач: {len(tasks)}")
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
    print(f"Извлечено символов: {len(result):,}")
    data_logger.info(f"Извлечено символов: {len(result):,}")
    
    if len(result) == 0:
        data_logger.warning("⚠️ Пустой результат конвертации задач!")
        print("⚠️ Внимание: задачи пустые или некорректный формат JSON")
    
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
    
    print(f"Загрузка JSON: вопросы для интервью")
    data_logger.info(f"JSON формат: интервью")
    print(f"Найдено вопросов: {len(items)}")
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
    print(f"Извлечено символов: {len(result):,}")
    data_logger.info(f"Извлечено символов: {len(result):,}")
    
    if len(result) == 0:
        data_logger.warning("⚠️ Пустой результат конвертации интервью!")
        print("⚠️ Внимание: вопросы пустые или некорректный формат JSON")
    
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
    include_topics: bool = True
) -> tuple[DataLoader, DataLoader, CharTokenizer]:
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
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    log_session_start(data_logger, "Загрузка данных")
    data_logger.info(f"Путь к данным: {data_path}")
    data_logger.info(f"Параметры: context_len={context_len}, batch_size={batch_size}, train_split={train_split}")
    
    # Определяем формат файла
    file_format = detect_file_format(data_path)
    print(f"Формат данных: {file_format}")
    data_logger.info(f"Формат данных: {file_format}")
    
    # Читаем текст
    if file_format == 'json':
        # Определяем тип JSON
        json_type = detect_json_type(data_path)
        print(f"Тип JSON: {json_type}")
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
    
    print(f"Общий размер текста: {len(text):,} символов")
    data_logger.info(f"Общий размер текста: {len(text):,} символов")
    
    if len(text) == 0:
        raise ValueError("Пустой текст после загрузки! Проверьте содержимое файла.")
    
    # Создаём токенизатор
    tokenizer = CharTokenizer(text)
    data_logger.info(f"Токенизатор создан: {len(tokenizer.vocab)} токенов")
    
    # Разбиваем на train/val
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Создаём датасеты
    train_dataset = TextDataset(train_text, context_len, tokenizer.vocab)
    val_dataset = TextDataset(val_text, context_len, tokenizer.vocab)
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
        print("⚠️ Validation датасет пуст — используем train для валидации")
        val_dataset = train_dataset  # Используем train как val
    
    print(f"✅ Train: {len(train_dataset)} окон, Val: {len(val_dataset)} окон")
    
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
    print(f"Создан sample датасет: {output_path}")
