# 🚀 Enterprise GPT Training Platform

**Обучите GPT-модель на корпоративных данных. On-premise. Конфиденциально. Без API.**

Production-ready платформа для обучения и деплоя специализированных языковых моделей на ваших данных.  
+ Полный образовательный пример построения LLM с нуля для изучения трансформеров.

---

## 🎯 Для кого этот проект?

### 💼 Enterprise & Production

- **🏢 ML-команды в компаниях** — обучите AI на внутренних данных
- **💼 Консалтинговые фирмы** — создайте domain-specific решения для клиентов
- **🚀 AI-стартапы** — быстрый MVP без зависимости от OpenAI API
- **🔒 Enterprise** — полный контроль и конфиденциальность данных

### 🎓 Образование & Обучение

- **📚 Студенты ML/DL** — изучите архитектуру трансформеров на практике
- **👨‍💻 Junior ML Engineers** — поймите, как устроены ChatGPT/GPT-4 изнутри
- **🔬 Исследователи** — экспериментируйте с модификациями архитектуры
- **👨‍🏫 Преподаватели** — используйте как учебный материал для курсов

---

## 📚 Use Cases

### 🏢 **Enterprise Applications**

#### Корпоративные AI-ассистенты
Обучите модель на внутренней документации компании для ответов на вопросы сотрудников. 
**Результат:** HR-боты, tech support, sales assistants на основе вашей базы знаний.

#### Domain-Specific модели
Специализированные модели для узких областей с профессиональной терминологией:
- **Медицина:** диагностика, анализ симптомов (HIPAA compliant)
- **Юриспруденция:** анализ контрактов, поиск прецедентов
- **Финансы:** анализ отчетов, compliance
- **Код:** генерация на внутренних фреймворках

#### Customer Support Automation
AI первой линии поддержки, обученный на истории тикетов и FAQ.
**Результат:** 60-80% типовых вопросов решаются автоматически.

#### Автоматизация документации
Генерация технической документации, API docs, release notes в стиле компании.

### 🎓 **Educational Use Cases**

#### Курсовые и дипломные работы
Полная реализация трансформера от attention механизма до генерации текста.
**Результат:** Понимание устройства современных LLM на практике.

#### Исследовательские эксперименты
Чистая, читаемая реализация для экспериментов:
- Тестирование новых attention механизмов
- Эксперименты с tokenization
- Сравнение архитектур
- Baseline для научных статей

#### Обучение команды
Onboarding новых ML-инженеров в компании:
- Понимание full ML pipeline
- Best practices в production
- Hands-on опыт с трансформерами

---

## ✨ Ключевые преимущества

### 🔒 Конфиденциальность (для Enterprise)
- ✅ Данные не покидают вашу инфраструктуру
- ✅ Соответствие GDPR, HIPAA, SOC2
- ✅ Нет vendor lock-in
- ✅ Offline работа

### 💰 Экономия (для Enterprise)
- ✅ $0 за API calls (vs $1,000-10,000/мес за OpenAI)
- ✅ Разовые затраты на обучение vs постоянные на API
- ✅ Масштабирование без дополнительных costs

### 🎯 Customization (для Enterprise)
- ✅ Обучение на терминологии вашей компании
- ✅ Специализация под узкий домен
- ✅ Меньше hallucinations (обучена на правильных данных)
- ✅ Fine-tuning под конкретные задачи

### 📚 Образовательная ценность
- ✅ **Полная прозрачность** — весь код открыт и читаем (~2000 строк)
- ✅ **Понятная архитектура** — чистая реализация без лишних абстракций
- ✅ **Production best practices** — early stopping, checkpointing, validation
- ✅ **Streamlit UI** — визуализация обучения в реальном времени
- ✅ **Детальное логирование** — отслеживание каждого шага

### 🚀 Production-Ready
- ✅ Полный training pipeline с best practices
- ✅ Model management (versioning, checkpointing)
- ✅ Streamlit UI для non-technical users
- ✅ Device optimization (MPS/CUDA/CPU)
- ✅ Comprehensive logging и monitoring

---

## 📁 Структура проекта

```
LLM-learn/
├── ocr/                    # OCR — распознавание текста (отдельный модуль)
│   ├── ocr_app.py          # Streamlit UI для Tesseract
│   └── data_preparation/   # Временные файлы сканов и задач
│
├── pytorch_llm/            # PyTorch LLM — обучение моделей (основной модуль)
│   ├── app.py              # Streamlit UI для обучения
│   ├── model.py            # GPT-архитектура
│   ├── training.py         # Training loop
│   ├── data.py             # Data loading
│   ├── inference.py        # Text generation
│   ├── data/               # Датасеты для обучения
│   └── checkpoints/        # Сохраненные модели
│
├── mini_llm/               # NumPy LLM — учебная реализация
│   └── v3/                 # Версия 3 (NumPy трансформер)
│
└── docs/                   # Документация
    ├── JSON_FORMATS.md
    └── DEVICE_MONITORING.md
```

---

## 🎯 Принципы архитектуры (SOLID)

### **Single Responsibility Principle:**

Каждый модуль отвечает за одну область:

| Модуль | Ответственность |
|--------|----------------|
| `ocr/` | 📷 **Распознавание** текста из изображений |
| `pytorch_llm/` | 🧠 **Обучение** языковых моделей |
| `mini_llm/` | 📚 **Учебная** реализация на NumPy |

**Преимущества:**
- ✅ Модули независимы друг от друга
- ✅ Можно использовать отдельно
- ✅ Проще тестировать и поддерживать
- ✅ Четкое разделение зависимостей

---

## 🚀 Быстрый старт

### **1. Установка:**

```bash
# Клонирование
git clone <repo-url>
cd LLM-learn

# Установка зависимостей
make install

# Установка Tesseract (для OCR)
brew install tesseract tesseract-lang
```

### **2. OCR — Распознавание текста:**

```bash
make tesseract
# Откроется: http://localhost:8503
```

**Workflow:**
1. Загрузите сканы учебников
2. Распознайте текст через Tesseract
3. Отредактируйте в WYSIWYG редакторе
4. Экспортируйте в JSON
5. Скопируйте в `pytorch_llm/data/`

**Документация:** [ocr/README.md](ocr/README.md)

---

### **3. PyTorch LLM — Обучение моделей:**

```bash
make pytorch
# Откроется: http://localhost:8502
```

**Workflow:**
1. Выберите режим (с нуля / fine-tuning)
2. Загрузите датасет (.txt или .json)
3. Настройте параметры обучения
4. Запустите обучение
5. Генерируйте текст

**Документация:** [pytorch_llm/README.md](pytorch_llm/README.md)

---

### **4. Mini LLM — Учебная реализация:**

```bash
make ui
# NumPy трансформер (для изучения)
```

---

## 📋 Основные команды

```bash
# OCR
make tesseract       # Запуск OCR приложения
make ocr             # Алиас для tesseract

# PyTorch LLM
make pytorch         # Запуск обучения моделей
make api             # API сервер (OpenAI-совместимый)
make inference       # CLI тестирование модели
make convert-json    # Конвертация JSON → TXT

# Mini LLM (NumPy)
make ui              # Streamlit UI
make cli             # Консольный интерфейс
make train           # Обучение и сохранение

# Утилиты
make install         # Установка зависимостей
make clean           # Очистка моделей и кэша
make help            # Полный список команд
```

---

## 🔄 Workflow: От скана до обученной модели

```
1. OCR (ocr/)
   ├─ Загрузка: scan.jpg → ocr/data_preparation/scans/
   ├─ Распознавание: Tesseract OCR
   └─ Экспорт: chemistry_ch1.json → ocr/data_preparation/tasks/

2. Копирование в LLM (вручную или кнопкой)
   └─ chemistry_ch1.json → pytorch_llm/data/

3. Обучение (pytorch_llm/)
   ├─ Режим: Fine-tuning
   ├─ Checkpoint: chemistry_model_best.pt
   ├─ Данные: chemistry_ch1.json
   └─ Результат: chemistry_model_updated.pt

4. Генерация
   └─ Промпт: "Какова молярная масса H₂O?"
```

---

## 🛠️ Технологии

### **OCR модуль:**
- Tesseract — распознавание текста
- Streamlit — веб-интерфейс
- Pillow — обработка изображений
- JavaScript — WYSIWYG редактор формул

### **PyTorch LLM модуль:**
- PyTorch 2.12+ — фреймворк
- GPT-архитектура — трансформер
- Apple MPS / CUDA / CPU — ускорение
- FastAPI — REST API
- Streamlit — веб-интерфейс

### **Mini LLM модуль:**
- NumPy — чистая математика
- Streamlit — учебный UI

---

## 📚 Документация

| Документ | Описание |
|----------|----------|
| [ocr/README.md](ocr/README.md) | OCR модуль (Tesseract) |
| [pytorch_llm/README.md](pytorch_llm/README.md) | PyTorch LLM модуль |
| [pytorch_llm/QUICKSTART.md](pytorch_llm/QUICKSTART.md) | Быстрый старт обучения |
| [docs/JSON_FORMATS.md](docs/JSON_FORMATS.md) | Форматы данных |
| [docs/DEVICE_MONITORING.md](docs/DEVICE_MONITORING.md) | Мониторинг GPU/MPS |
| [Makefile](Makefile) | Все команды |

---

## 🎓 Для чего этот проект?

### **Обучение:**
- 📚 Понять, как работают трансформеры
- 🔬 Изучить архитектуру GPT
- 💻 Попрактиковать PyTorch
- 🧮 Реализовать на чистом NumPy

### **Практика:**
- 🤖 Обучить собственную языковую модель
- 📝 Подготовить данные из учебников (OCR)
- 🎯 Fine-tuning на специфических данных
- 🚀 Развернуть API для генерации

---

## ⚙️ Зависимости

**Python пакеты:**
```toml
[tool.poetry.dependencies]
python = ">=3.12"
torch = ">=2.0.0"
streamlit = ">=1.45.0"
fastapi = ">=0.136.3"
pytesseract = ">=0.3.13"
pillow = ">=12.2.0"
numpy = ">=2.4.6"
# ... и другие
```

**Системные зависимости:**
- Tesseract OCR (macOS: `brew install tesseract tesseract-lang`)

---

## 🤝 Разработка

### **Добавление новых данных:**

```bash
# 1. OCR новых страниц
make tesseract

# 2. Объединение с существующими данными
cd pytorch_llm/data
cat dataset_old.txt new_data.txt > dataset_combined.txt

# 3. Fine-tuning модели
make pytorch
→ Режим: Fine-tuning
→ Checkpoint: model_step_2500.pt
→ Данные: dataset_combined.txt
```

### **Структура checkpoint:**

```python
checkpoint = {
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'config': ...,
    'vocab': ...,           # Словарь токенизатора
    'global_step': 2500,
    'best_val_loss': 1.92,
    'model_name': 'chemistry_model',
    'training_date': '2026-06-01T10:15:06',
}
```

---

## 🧹 Очистка

```bash
# Удалить модели
rm -rf pytorch_llm/checkpoints/*.pt

# Очистить OCR временные файлы
rm -rf ocr/data_preparation/scans/*
rm -rf ocr/data_preparation/tasks/*

# Или через make
make clean
```

---

## 📊 Статистика

- **NumPy реализация:** ~500 строк кода
- **PyTorch реализация:** ~3,000 строк кода
- **OCR модуль:** ~1,200 строк кода
- **Поддержка форматов:** TXT, JSON (tasks, forum, interview)
- **Устройства:** MPS, CUDA, CPU

---

## 🎯 Roadmap

- [x] NumPy трансформер
- [x] PyTorch GPT-архитектура
- [x] Training с early stopping
- [x] Fine-tuning
- [x] OCR модуль (Tesseract)
- [x] WYSIWYG редактор формул
- [x] Множественные форматы JSON
- [x] Device мониторинг (MPS/CUDA)
- [x] Модульная архитектура (SOLID)
- [ ] Тесты (pytest)
- [ ] Docker контейнеры
- [ ] Continual Learning

---

**Автор:** Senior Python ML Engineer  
**Дата:** 2026-06-01  
**Лицензия:** MIT  
**Принципы:** Clean Code, SOLID, DRY
