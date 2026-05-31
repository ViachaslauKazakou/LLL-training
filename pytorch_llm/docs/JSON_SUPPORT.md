# JSON Data Support

## Обзор

PyTorch LLM теперь поддерживает обучение на JSON данных с форумными сообщениями. Система автоматически определяет формат файла и конвертирует JSON в текст для обучения.

## Формат JSON

Ожидаемая структура JSON файла:

```json
{
  "User": "user_id",
  "messages": {
    "Топик 1": [
      "Сообщение 1",
      "Сообщение 2",
      ...
    ],
    "Топик 2": [
      "Сообщение 1",
      ...
    ]
  }
}
```

## Использование

### 1. Через CLI

```bash
# Обучение с нуля на JSON данных
poetry run python train.py \
  --data data/forum_messages.json \
  --config small \
  --epochs 10

# Дообучение на JSON
poetry run python train.py \
  --continue-from checkpoints/best_model.pt \
  --data data/new_messages.json \
  --epochs 5

# Без названий топиков
poetry run python train.py \
  --data data/forum_messages.json \
  --no-include-topics
```

### 2. Через Python API

```python
from data import load_data, convert_forum_json_to_text

# Конвертация JSON -> текст
text = convert_forum_json_to_text(
    'data/forum_messages.json',
    include_topics=True  # Включить названия топиков
)

# Загрузка данных для обучения
train_loader, val_loader, tokenizer = load_data(
    'data/forum_messages.json',
    context_len=256,
    batch_size=32,
    include_topics=True
)
```

### 3. Через Streamlit UI

1. Откройте вкладку **📁 Подготовка данных**
2. Загрузите JSON файл через file uploader
3. Система автоматически:
   - Распознает JSON формат
   - Покажет статистику (топики, сообщения, символы)
   - Предложит сохранить как .txt

**Конвертация существующих JSON:**
- Выберите JSON файл из списка
- Настройте опцию "Включить названия топиков"
- Нажмите "🔄 Конвертировать"

### 4. Обучение в UI

В вкладке **🎓 Обучение**:
- Укажите путь к JSON файлу в поле "Путь к данным"
- Система автоматически определит формат и загрузит данные
- Названия топиков включаются по умолчанию

## Параметры

### `include_topics` (bool, default=True)

Контролирует, добавлять ли названия топиков как контекст:

**С топиками (include_topics=True):**
```
### Что может, что не может AI

Вашей оголтелой кучке Софтерра...

Уже сами погромисты ваши жалуются...
```

**Без топиков (include_topics=False):**
```
Вашей оголтелой кучке Софтерра...

Уже сами погромисты ваши жалуются...
```

## Примеры

### Обучение на форумных данных

```bash
# 1. Проверить данные
poetry run python analyze_json.py

# 2. Обучить small модель
poetry run python train.py \
  --data data/parser_2039898_20260521T071908.json \
  --config small \
  --epochs 30 \
  --batch-size 16 \
  --lr 3e-4

# 3. Дообучить на новых сообщениях
poetry run python train.py \
  --continue-from checkpoints/best_model.pt \
  --data data/new_forum_posts.json \
  --epochs 10 \
  --lr 1e-4
```

### Генерация текста

```bash
# После обучения
poetry run python inference.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "Искусственный интеллект" \
  --max-tokens 200 \
  --interactive
```

## Статистика данных

Пример файла `parser_2039898_20260521T071908.json`:
- Размер: 253.5 KB
- Пользователь: 2039898
- Топиков: 112
- Сообщений: 660
- Символов: ~140K
- Токенов (char-level): ~127K

## Производительность

Small модель (3.2M параметров) на JSON данных:
- ~35 it/s на MPS (Apple Silicon M1/M2)
- ~15K шагов на эпоху
- Validation loss: ~2.68 после 500 шагов

## Технические детали

### Автоопределение формата

```python
def detect_file_format(data_path: str) -> str:
    """Определяет формат по расширению."""
    suffix = Path(data_path).suffix.lower()
    return 'json' if suffix == '.json' else 'text'
```

### Конвертация

```python
def convert_forum_json_to_text(json_path: str, include_topics: bool) -> str:
    """Конвертирует JSON с сообщениями в текст."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text_parts = []
    for topic, msgs in data['messages'].items():
        if include_topics:
            text_parts.append(f"\n\n### {topic}\n\n")
        for msg in msgs:
            text_parts.append(msg.strip() + '\n\n')
    
    return ''.join(text_parts)
```

## Рекомендации

1. **Размер данных:** Минимум 100K символов, оптимально 1M+
2. **include_topics:** Включайте для сохранения контекста
3. **Batch size:** Уменьшите для JSON (больше разнообразия на топик)
4. **Эпохи:** 20-50 эпох для хорошей сходимости
5. **Learning rate:** 3e-4 для обучения с нуля, 1e-4 для fine-tuning

## Ограничения

- Поддерживается только char-level токенизация
- JSON должен иметь структуру `{"User": ..., "messages": {...}}`
- Максимальная длина контекста зависит от конфигурации модели
- Очень длинные сообщения не обрезаются автоматически
