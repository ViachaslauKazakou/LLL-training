# 🚀 Быстрый запуск

## PyTorch Transformer

### Тренировка модели (Streamlit UI)
```bash
make transformer
```
Откроется веб-интерфейс на http://localhost:8501

### OCR приложение (подготовка данных)
```bash
make ocr
```
Откроется OCR интерфейс на http://localhost:8503

**Workflow:**
1. Загрузите сканы учебников
2. Распознайте текст (Tesseract или GPT-4 Vision)
3. Отредактируйте с помощью панели символов (H₂O, 10²³, →)
4. Извлеките задачи
5. Сохраните в JSON для тренировки

### API Сервер (OpenAI-совместимый)
```bash
make api
```
API будет доступен на http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Эндпоинты: `/v1/chat/completions`, `/v1/completions`, `/v1/models`

### CLI Inference (быстрое тестирование)
```bash
make inference
```

---

## Полный список команд

Смотрите все доступные команды:
```bash
make help
```

Или просто:
```bash
make
```

---

## Установка зависимостей

Первый раз после клонирования:
```bash
make install
```

---

## Типичные сценарии

### 1. Тренировка новой модели
```bash
# Шаг 1: Подготовьте данные
make ocr
# → Загрузите сканы, распознайте, сохраните JSON

# Шаг 2: Запустите тренировку
make transformer
# → Загрузите JSON, настройте параметры, запустите обучение
```

### 2. Использование обученной модели
```bash
# Вариант 1: API сервер для интеграции
make api

# Вариант 2: CLI для быстрых тестов
make inference
```

### 3. Разработка и тестирование
```bash
# Окно 1: Запустите трансформер
make transformer

# Окно 2: Запустите OCR параллельно
make ocr

# Окно 3: Тестируйте модель
make inference
```

---

## Порты

| Приложение | Порт | Команда |
|------------|------|---------|
| Трансформер UI | 8501 | `make transformer` |
| OCR UI | 8503 | `make ocr` |
| API сервер | 8000 | `make api` |

---

## Очистка

Удалить кэш и старые логи:
```bash
make clean
```

---

## Структура проекта

```
LLM-learn/
├── pytorch_llm/          # PyTorch трансформер
│   ├── app.py           # UI для тренировки
│   ├── ocr_app.py       # OCR приложение (JS редактор)
│   ├── api_server.py    # OpenAI-совместимый API
│   ├── model.py         # GPT-style transformer
│   ├── training.py      # Trainer с Early Stopping
│   └── data/            # Данные для тренировки
├── mini_llm/            # Оригинальная NumPy версия
└── Makefile             # ← Команды быстрого запуска
```
