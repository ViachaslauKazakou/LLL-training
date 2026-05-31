# Система логирования PyTorch LLM

## Обзор

Все ключевые операции автоматически логируются в файлы для мониторинга и отладки.

## Файлы логов

Все логи сохраняются в директории `logs/`:

| Файл | Описание |
|------|----------|
| `data.log` | Загрузка данных, токенизация, создание датасетов |
| `training.log` | Процесс обучения, метрики, сохранение checkpoints |
| `inference.log` | Генерация текста, параметры, результаты |
| `app.log` | События UI (Streamlit), действия пользователя |

## Формат логов

```
2026-05-30 19:27:31 [INFO] Загрузка данных — НАЧАЛО СЕССИИ
2026-05-30 19:27:31 [INFO] Время: 2026-05-30 19:27:31
2026-05-30 19:27:31 [INFO] ============================================================
2026-05-30 19:27:31 [INFO] Путь к данным: data/sample.txt
2026-05-30 19:27:31 [INFO] Параметры: context_len=64, batch_size=4, train_split=0.9
2026-05-30 19:27:31 [INFO] Формат данных: text
2026-05-30 19:27:31 [INFO] Загружен текстовый файл: 8,750 символов
2026-05-30 19:27:31 [INFO] ============================================================
2026-05-30 19:27:31 [INFO] Загрузка данных — КОНЕЦ СЕССИИ
2026-05-30 19:27:31 [INFO] Время: 2026-05-30 19:27:31
2026-05-30 19:27:31 [INFO] ============================================================
```

### Структура записи

- **Timestamp**: `2026-05-30 19:27:31`
- **Level**: `[INFO]`, `[WARNING]`, `[ERROR]`
- **Message**: Текст события

## Что логируется

### 📁 Загрузка данных (data.log)

- Путь к файлу данных
- Формат файла (.txt или .json)
- Размер данных (символы)
- Количество токенов
- Размеры train/val датасетов
- Количество батчей

**Пример:**
```
[INFO] Загрузка данных — НАЧАЛО СЕССИИ
[INFO] Путь к данным: data/forum_messages.json
[INFO] Формат данных: json
[INFO] JSON пользователь: 2039898
[INFO] Найдено топиков: 112
[INFO] Всего сообщений: 660
[INFO] Извлечено символов: 140,685
[INFO] Токенизатор создан: 171 токенов
[INFO] Датасеты созданы: train=126616, val=14069
[INFO] DataLoader'ы созданы: train_batches=15795, val_batches=1747
[INFO] Загрузка данных — КОНЕЦ СЕССИИ
```

### 🎓 Обучение (training.log)

- Параметры обучения (эпохи, batch size, learning rate)
- Количество параметров модели
- Устройство (MPS/CUDA/CPU)
- Метрики по эпохам (train loss, val loss, perplexity)
- Сохранение checkpoints
- Лучший val_loss

**Пример:**
```
[INFO] Обучение модели — НАЧАЛО СЕССИИ
[INFO] Устройство: mps
[INFO] Параметров модели: 3,268,864
[INFO] Эпох: 10
[INFO] Batch size: 16
[INFO] Learning rate: 0.0003
[INFO] Epoch 1/10 завершена
[INFO]   Train: loss=2.5432, perplexity=12.71
[INFO]   Val:   loss=2.3145, perplexity=10.12
[INFO] Сохранён лучший checkpoint: val_loss=2.3145 (step 500)
[INFO] Сохранён periodic checkpoint: checkpoint_step_1000.pt
[INFO] Обучение завершено!
[INFO] Лучший val_loss: 2.1234
[INFO] Обучение модели — КОНЕЦ СЕССИИ
```

### 💬 Генерация (inference.log)

- Загрузка checkpoint
- Параметры модели
- Промпты
- Параметры генерации (temperature, top_k, max_tokens)
- Размер сгенерированного текста

**Пример:**
```
[INFO] Загрузка checkpoint: checkpoints/best_model.pt
[INFO] Используемое устройство: mps
[INFO] Модель загружена: 3,268,864 параметров
[INFO] Global step: 15795
[INFO] Best val loss: 2.1234
[INFO] Генерация текста
[INFO] Промпт: 'Искусственный интеллект' (26 символов)
[INFO] Параметры: max_tokens=100, temp=0.8, top_k=50
[INFO] Сгенерировано: 487 символов
```

### 🖥️ UI События (app.log)

- Запуск обучения через UI
- Параметры из формы
- Генерация через UI
- Ошибки

**Пример:**
```
[INFO] UI Обучение запущено: mode=from scratch
[INFO] UI Параметры: epochs=30, batch_size=16, lr=0.0003, data=data/forum.json
[INFO] UI Обучение завершено успешно
[INFO] UI Генерация запущена: prompt='Искусственный...', max_tokens=100, temp=0.8, top_k=50
[INFO] UI Генерация завершена: 487 символов
[ERROR] UI Ошибка при обучении: CUDA out of memory
```

## Использование

### Автоматическое логирование

Логирование работает автоматически при использовании CLI или UI:

```bash
# CLI обучение
poetry run python train.py --data data/sample.txt --epochs 10
# → Логи в logs/data.log и logs/training.log

# CLI генерация
poetry run python inference.py --checkpoint best_model.pt --prompt "Текст"
# → Логи в logs/inference.log

# UI
poetry run streamlit run app.py
# → Логи в logs/app.log, logs/data.log, logs/training.log, logs/inference.log
```

### Программное использование

```python
from logger import data_logger, log_session_start, log_session_end

# Начало сессии
log_session_start(data_logger, "Моя операция")

# Логирование событий
data_logger.info("Загружаю данные...")
data_logger.info(f"Размер: {size} байт")

# Предупреждения
data_logger.warning("Файл может быть поврежден")

# Ошибки
try:
    load_data()
except Exception as e:
    data_logger.error(f"Ошибка загрузки: {e}")

# Конец сессии
log_session_end(data_logger, "Моя операция")
```

### Доступные логгеры

```python
from logger import (
    data_logger,       # Для загрузки данных
    training_logger,   # Для обучения
    inference_logger,  # Для генерации
    app_logger,        # Для UI
    log_session_start, # Начало сессии
    log_session_end    # Конец сессии
)
```

## Настройка

### Изменить уровень логирования

В `logger.py`:

```python
# По умолчанию: logging.INFO
data_logger = setup_logger("pytorch_llm.data", "data.log", level=logging.DEBUG)
```

Уровни:
- `DEBUG` — все детали
- `INFO` — основная информация (по умолчанию)
- `WARNING` — только предупреждения
- `ERROR` — только ошибки

### Изменить формат

В `logger.py`:

```python
format_string = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
```

### Ротация логов

Для автоматической очистки старых логов используйте `RotatingFileHandler`:

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    log_path,
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5            # Хранить 5 файлов
)
```

## Просмотр логов

### В терминале

```bash
# Просмотр полного лога
cat logs/training.log

# Последние 50 строк
tail -50 logs/training.log

# Следить за логом в реальном времени
tail -f logs/training.log

# Поиск по логам
grep "ERROR" logs/*.log
grep "val_loss" logs/training.log
```

### В Python

```python
from pathlib import Path

# Прочитать лог
log_content = Path("logs/training.log").read_text()

# Последние N строк
lines = log_content.splitlines()
last_lines = lines[-50:]

# Поиск ошибок
errors = [line for line in lines if "[ERROR]" in line]
```

## Мониторинг

### Real-time мониторинг обучения

```bash
# В отдельном терминале во время обучения
tail -f logs/training.log | grep -E "(Epoch|val_loss|checkpoint)"
```

### Анализ после обучения

```bash
# Все метрики
grep -E "(Train|Val):" logs/training.log

# Только лучшие checkpoints
grep "Сохранён лучший" logs/training.log

# Все ошибки
grep "\[ERROR\]" logs/*.log
```

## Очистка логов

```bash
# Удалить все логи
rm logs/*.log

# Удалить старые логи (старше 7 дней)
find logs/ -name "*.log" -mtime +7 -delete

# Архивировать старые логи
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

## Примеры реальных сессий

### Полный цикл обучения

```
logs/data.log:
  [INFO] Загрузка данных — НАЧАЛО СЕССИИ
  [INFO] Путь к данным: data/forum.json
  [INFO] Формат данных: json
  [INFO] Общий размер текста: 140,685 символов
  [INFO] Загрузка данных — КОНЕЦ СЕССИИ

logs/training.log:
  [INFO] Обучение модели — НАЧАЛО СЕССИИ
  [INFO] Устройство: mps
  [INFO] Параметров модели: 3,268,864
  [INFO] Эпох: 30
  [INFO] Epoch 1/30 завершена
  [INFO]   Train: loss=4.2145, perplexity=67.51
  [INFO]   Val:   loss=3.9821, perplexity=53.61
  ...
  [INFO] Epoch 30/30 завершена
  [INFO]   Train: loss=1.8532, perplexity=6.38
  [INFO]   Val:   loss=2.0145, perplexity=7.50
  [INFO] Обучение завершено!
  [INFO] Обучение модели — КОНЕЦ СЕССИИ

logs/inference.log:
  [INFO] Загрузка checkpoint: checkpoints/best_model.pt
  [INFO] Модель загружена: 3,268,864 параметров
  [INFO] Генерация текста
  [INFO] Промпт: 'Искусственный интеллект'
  [INFO] Сгенерировано: 523 символов
```

## Troubleshooting

### Логи не создаются

Проверьте права доступа:
```bash
mkdir -p logs
chmod 755 logs
```

### Логи слишком большие

Настройте ротацию или очистите старые:
```bash
# Размер логов
du -h logs/

# Очистить logs старше 7 дней
find logs/ -name "*.log" -mtime +7 -delete
```

### Не видно ошибок в консоли

Console handler показывает только WARNING и выше. Для DEBUG/INFO смотрите файлы логов.
