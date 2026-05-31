# Логирование — Quick Start

## Где находятся логи

```
pytorch_llm/logs/
├── data.log       # Загрузка данных
├── training.log   # Обучение
├── inference.log  # Генерация текста
└── app.log        # UI события
```

## Автоматическое логирование

Все операции логируются автоматически:

```bash
# Обучение
poetry run python train.py --data data/sample.txt --epochs 10
# → Логи в data.log и training.log

# Генерация
poetry run python inference.py --checkpoint best_model.pt --prompt "Текст"
# → Логи в inference.log

# UI
poetry run streamlit run app.py
# → Логи во всех файлах
```

## Просмотр логов

```bash
# Весь лог
cat logs/training.log

# Последние строки
tail -20 logs/training.log

# В реальном времени
tail -f logs/training.log

# Поиск ошибок
grep ERROR logs/*.log
```

## Формат

```
2026-05-30 19:29:40 [INFO] Обучение модели — НАЧАЛО СЕССИИ
2026-05-30 19:29:40 [INFO] Устройство: mps
2026-05-30 19:29:40 [INFO] Параметров модели: 3,236,096
2026-05-30 19:29:40 [INFO] Эпох: 10
2026-05-30 19:29:40 [INFO] Batch size: 16
...
2026-05-30 19:29:40 [INFO] Epoch 1/10 завершена
2026-05-30 19:29:40 [INFO]   Train: loss=2.5432, perplexity=12.71
2026-05-30 19:29:40 [INFO]   Val:   loss=2.3145, perplexity=10.12
...
2026-05-30 19:29:40 [INFO] Обучение завершено!
2026-05-30 19:29:40 [INFO] Обучение модели — КОНЕЦ СЕССИИ
```

Полная документация: [LOGGING.md](./LOGGING.md)
