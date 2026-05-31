# 🎓 План реализации AI-репетитора по химии (Hybrid подход)

**Дата создания:** 30 мая 2026  
**Цель:** Создать экономичный AI-репетитор, использующий собственную модель для типовых задач (70% запросов) и GPT-4 для сложных случаев (30% запросов)

---

## 📋 Содержание

1. [Обзор проблемы и решения](#обзор-проблемы-и-решения)
2. [Архитектура системы](#архитектура-системы)
3. [Технические детали](#технические-детали)
4. [План реализации](#план-реализации)
5. [Подготовка данных](#подготовка-данных)
6. [Обучение модели](#обучение-модели)
7. [Создание Router](#создание-router)
8. [Интеграция и развертывание](#интеграция-и-развертывание)
9. [Метрики и оптимизация](#метрики-и-оптимизация)
10. [Альтернативные подходы](#альтернативные-подходы)

---

## 🎯 Обзор проблемы и решения

### Проблема
- **Текущая ситуация:** AI-репетитор по химии работает только на GPT-4/Claude
- **Стоимость:** ~$60/месяц при 30,000 запросов
- **Проблемы:**
  - 💰 Высокие расходы на токены для простых задач
  - ⏱️ Задержки 1-2 секунды на каждый запрос
  - 🔓 Данные учеников уходят на серверы OpenAI
  - 📈 Масштабирование дорого

### Решение: Hybrid AI Architecture

**Идея:** Использовать собственную специализированную модель для типовых задач + GPT-4 для сложных случаев

**Выгоды:**
- 💰 **Экономия 70%:** $60/мес → $18/мес (экономия $504/год)
- ⚡ **Скорость:** 50-200ms локально vs 1-2s через API (в 4-40 раз быстрее)
- 🔒 **Приватность:** Простые запросы обрабатываются локально
- 🎯 **Специализация:** Модель обучена конкретно на химии
- 📊 **Контроль:** Полный контроль над моделью и её поведением

**Распределение нагрузки:**
- 70% запросов → Локальная модель (типовые задачи, формулы, уравнения)
- 30% запросов → GPT-4 (сложные объяснения, творческие задачи)

---

## 🏗️ Архитектура системы

### Высокоуровневая схема

```
┌─────────────────────────────────────────────────────────────┐
│                     Клиентское приложение                    │
│                   (Web/Mobile/Desktop)                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP/REST
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     FastAPI Backend                          │
│                  (chemistry_router.py)                       │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         🧠 Classification Logic                     │    │
│  │                                                     │    │
│  │  • Анализ сложности вопроса                        │    │
│  │  • Определение категории (формулы/объяснения)      │    │
│  │  • Выбор модели (local vs GPT-4)                   │    │
│  └──────────────────┬─────────────┬────────────────────┘    │
│                     │             │                         │
└─────────────────────┼─────────────┼─────────────────────────┘
                      │             │
        ┌─────────────▼───┐     ┌───▼─────────────┐
        │  🆓 LOCAL API   │     │  💰 OpenAI API  │
        │  (port 8000)    │     │   (GPT-4)       │
        │                 │     │                 │
        │ Custom LLM API  │     │ cloud-based     │
        │ (uvicorn)       │     │                 │
        └────────┬────────┘     └─────────────────┘
                 │
        ┌────────▼────────┐
        │  Ваша модель    │
        │  (PyTorch)      │
        │  best_model.pt  │
        └─────────────────┘
```

### Поток данных

```
Запрос ученика
    │
    ▼
┌─────────────────────────────────────────┐
│  1. Получение вопроса                   │
│     "Закончите уравнение: 2H₂ + O₂ → ?"│
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  2. Классификация (Router)              │
│                                          │
│  • Простые паттерны?                    │
│  • Длина текста?                        │
│  • Ключевые слова?                      │
│                                          │
│  Решение: easy → local model ✅         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  3. Обработка (Local API)               │
│                                          │
│  • Токенизация: текст → числа          │
│  • Инференс: model.generate()          │
│  • Декодирование: числа → текст        │
│                                          │
│  Ответ: "2H₂O" (время: 50ms)           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  4. Форматирование ответа               │
│                                          │
│  {                                       │
│    "answer": "2H₂O",                    │
│    "model": "local",                    │
│    "cost": 0.0,                         │
│    "latency_ms": 50                     │
│  }                                       │
└────────────────┬────────────────────────┘
                 │
                 ▼
        Ответ ученику ✅
```

### Компоненты системы

#### 1. **Local LLM API** (api_server.py)
- **Функция:** REST API для локальной модели
- **Технологии:** FastAPI, PyTorch, uvicorn
- **Порт:** 8000
- **Эндпоинты:**
  - `/v1/chat/completions` — OpenAI-compatible chat
  - `/v1/completions` — Text completion
  - `/v1/models` — Список моделей
  - `/health` — Health check

#### 2. **Router** (chemistry_router.py)
- **Функция:** Интеллектуальная маршрутизация запросов
- **Логика:**
  - Анализирует вопрос
  - Определяет сложность (easy/medium/hard)
  - Выбирает модель (local vs GPT-4)
  - Собирает метрики (стоимость, время)

#### 3. **Локальная модель** (checkpoints/best_model.pt)
- **Тип:** GPT-style transformer
- **Размер:** 3M-85M параметров (зависит от конфигурации)
- **Специализация:** Химические задачи, формулы, уравнения
- **Обучение:** На вашем датасете задач с решениями

#### 4. **Backend API** (app_chemistry.py)
- **Функция:** Основной бэкенд приложения
- **Технологии:** FastAPI
- **Задачи:**
  - Аутентификация учеников
  - История диалогов
  - Аналитика и метрики
  - Интеграция с Router

---

## 🔧 Технические детали

### Токенизация: CharTokenizer vs tiktoken

#### **CharTokenizer (текущая реализация)**

**Принцип работы:**
```python
# Каждый символ = отдельный токен
text = "H₂SO₄"
tokens = ['H', '₂', 'S', 'O', '₄']  # 5 токенов
ids = [45, 102, 78, 89, 103]
```

**Характеристики:**
- ✅ Простота реализации
- ✅ Не требует предобучения
- ✅ Маленький vocab (~100-200 символов)
- ❌ Длинные последовательности
- ❌ Не понимает семантику слов
- ❌ Менее эффективно для больших текстов

**Когда использовать:**
- Текущая модель (уже обучена с CharTokenizer)
- Быстрые прототипы
- Специфичные домены с ограниченным словарем

#### **tiktoken (BPE токенизация)**

**Принцип работы:**
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

text = "Машинное обучение"
tokens = enc.encode(text)  # [19333, 35431, 103457, 28976]
# Всего 4 токена (vs 17 с CharTokenizer)
```

**Характеристики:**
- ✅ Эффективная компрессия текста (меньше токенов)
- ✅ Понимает подслова и целые слова
- ✅ Универсальный vocab (~50k-100k токенов)
- ✅ Совместим с GPT-3/GPT-4 токенизатором
- ❌ Больше параметров (vocab_size 200 → 100k)
- ❌ Требует переобучения модели с нуля

**Когда использовать:**
- Новая модель (обучение с нуля)
- Production проекты
- Многоязычность
- Когда важна эффективность

#### **⚠️ Важно: Нельзя заменить токенизатор в обученной модели!**

```python
# ❌ ОШИБКА: Модель обучена с vocab_size=171
model = GPTModel(vocab_size=171, ...)  # CharTokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")  # vocab_size=100k
# → Несовместимость! Модель сломается!

# ✅ ПРАВИЛЬНО: Используйте тот же токенизатор
tokenizer = CharTokenizer(training_data)  # vocab_size=171
model = GPTModel(vocab_size=171, ...)
```

**Рекомендация:**
- **Для текущей модели:** Оставьте CharTokenizer
- **Для новой модели (v2):** Используйте tiktoken

---

## 📝 План реализации

### Фаза 0: Подготовка (1-2 дня)

#### ✅ Что уже есть:
- [x] PyTorch трансформер реализован
- [x] Обучена базовая модель (best_model.pt)
- [x] OpenAI-compatible API (api_server.py)
- [x] Streamlit UI для обучения
- [x] Инфраструктура логирования
- [x] Документация

#### ☐ Что нужно сделать:
- [ ] Собрать датасет химических задач
- [ ] Установить дополнительные зависимости
- [ ] Настроить окружение для production

```bash
# Установка зависимостей
cd /Users/Viachaslau_Kazakou/Work/LLM-learn/pytorch_llm
poetry add openai  # Для GPT-4 интеграции
poetry add redis   # Для кеширования (опционально)
poetry add prometheus-client  # Для метрик (опционально)
```

---

### Фаза 1: Подготовка данных (2-5 дней)

#### 1.1. Сбор данных

**Источники химических задач:**

1. **Сборники задач** (есть у вас):
   - Отсканируйте/конвертируйте в текст
   - Формат: задача + решение + ответ

2. **Онлайн источники:**
   - Открытые образовательные платформы
   - Учебники с решениями
   - Архивы олимпиад

3. **Синтетические данные:**
   - Генерация вариантов типовых задач
   - Использование шаблонов

**Целевой объем:**
- Минимум: 1,000 задач
- Оптимально: 5,000 задач
- Идеально: 20,000+ задач

#### 1.2. Структура данных

**Формат JSON:**

```json
{
  "dataset_name": "chemistry_tasks_ru",
  "version": "1.0",
  "created_at": "2026-05-30",
  "total_tasks": 5000,
  "categories": [
    "уравнения_реакций",
    "расчеты_по_формулам",
    "валентность",
    "молярная_масса",
    "стехиометрия",
    "органика",
    "неорганика"
  ],
  "tasks": [
    {
      "id": 1,
      "category": "уравнения_реакций",
      "difficulty": "easy",
      "question": "Закончите уравнение реакции: 2H₂ + O₂ → ?",
      "solution": "При соединении водорода с кислородом образуется вода. Уравнение: 2H₂ + O₂ → 2H₂O",
      "answer": "2H₂O",
      "explanation": "Два моля водорода реагируют с одним молем кислорода, образуя два моля воды",
      "keywords": ["водород", "кислород", "вода", "реакция соединения"],
      "tags": ["8_класс", "базовый_уровень"]
    },
    {
      "id": 2,
      "category": "молярная_масса",
      "difficulty": "easy",
      "question": "Рассчитайте молярную массу серной кислоты H₂SO₄",
      "solution": "M(H₂SO₄) = 2×M(H) + M(S) + 4×M(O) = 2×1 + 32 + 4×16 = 2 + 32 + 64 = 98 г/моль",
      "answer": "98 г/моль",
      "explanation": "Складываем атомные массы всех элементов с учётом индексов",
      "keywords": ["молярная масса", "серная кислота", "расчеты"],
      "tags": ["8_класс", "расчеты"]
    },
    {
      "id": 3,
      "category": "стехиометрия",
      "difficulty": "medium",
      "question": "Сколько граммов воды образуется при сгорании 4 г водорода?",
      "solution": "1. Уравнение: 2H₂ + O₂ → 2H₂O\n2. n(H₂) = m/M = 4/2 = 2 моль\n3. По уравнению: n(H₂O) = n(H₂) = 2 моль\n4. m(H₂O) = n×M = 2×18 = 36 г",
      "answer": "36 г",
      "explanation": "Используем стехиометрию и молярные массы",
      "keywords": ["стехиометрия", "расчеты по уравнению", "моли"],
      "tags": ["9_класс", "средняя_сложность"]
    },
    {
      "id": 4,
      "category": "органика",
      "difficulty": "hard",
      "question": "Предложите механизм реакции нуклеофильного замещения в CH₃Br с OH⁻",
      "solution": "Механизм SN2:\n1. Нуклеофил OH⁻ атакует атом углерода с обратной стороны от Br\n2. Образуется переходное состояние с пентакоординированным углеродом\n3. Происходит одновременное образование связи C-OH и разрыв C-Br\n4. Продукт: CH₃OH + Br⁻\n5. Конфигурация инвертируется (обращение Вальдена)",
      "answer": "CH₃OH + Br⁻ (механизм SN2 с обращением конфигурации)",
      "explanation": "Бимолекулярное нуклеофильное замещение с обращением конфигурации",
      "keywords": ["органика", "механизм реакции", "SN2", "нуклеофильное замещение"],
      "tags": ["11_класс", "высокая_сложность", "механизмы"]
    }
  ]
}
```

#### 1.3. Скрипт конвертации данных

Создайте `data_preparation/convert_chemistry_data.py`:

```python
"""
Конвертация химических задач в формат для обучения модели.
"""

import json
from pathlib import Path
from typing import List, Dict

def load_json_dataset(json_path: str) -> Dict:
    """Загружает датасет из JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_to_training_format(
    tasks: List[Dict],
    include_solution: bool = True,
    include_explanation: bool = False
) -> str:
    """
    Конвертирует задачи в текстовый формат для обучения.
    
    Args:
        tasks: Список задач
        include_solution: Включать подробное решение
        include_explanation: Включать объяснение
    """
    output = []
    
    for task in tasks:
        # Формат: Question → Solution → Answer
        lines = []
        lines.append(f"Задача: {task['question']}")
        
        if include_solution and 'solution' in task:
            lines.append(f"Решение: {task['solution']}")
        
        lines.append(f"Ответ: {task['answer']}")
        
        if include_explanation and 'explanation' in task:
            lines.append(f"Пояснение: {task['explanation']}")
        
        output.append('\n'.join(lines))
        output.append('')  # Пустая строка между задачами
    
    return '\n'.join(output)

def split_by_difficulty(tasks: List[Dict]) -> Dict[str, List[Dict]]:
    """Разделяет задачи по уровню сложности."""
    result = {
        'easy': [],
        'medium': [],
        'hard': []
    }
    
    for task in tasks:
        difficulty = task.get('difficulty', 'medium')
        result[difficulty].append(task)
    
    return result

def create_training_files(
    input_json: str,
    output_dir: str = 'data/chemistry'
):
    """
    Создаёт файлы для обучения из JSON датасета.
    
    Создаёт файлы:
    - chemistry_all.txt — все задачи
    - chemistry_easy.txt — простые задачи
    - chemistry_medium.txt — средние задачи
    - chemistry_hard.txt — сложные задачи
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Загружаем датасет
    dataset = load_json_dataset(input_json)
    tasks = dataset['tasks']
    
    print(f"📊 Загружено задач: {len(tasks)}")
    
    # Все задачи
    all_text = convert_to_training_format(tasks, include_solution=True)
    (output_path / 'chemistry_all.txt').write_text(all_text, encoding='utf-8')
    print(f"✅ Создан файл: chemistry_all.txt ({len(all_text)} символов)")
    
    # Разделяем по сложности
    by_difficulty = split_by_difficulty(tasks)
    
    for difficulty, difficulty_tasks in by_difficulty.items():
        if difficulty_tasks:
            text = convert_to_training_format(difficulty_tasks, include_solution=True)
            filename = f'chemistry_{difficulty}.txt'
            (output_path / filename).write_text(text, encoding='utf-8')
            print(f"✅ Создан файл: {filename} ({len(difficulty_tasks)} задач)")
    
    # Статистика по категориям
    categories = {}
    for task in tasks:
        cat = task.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📈 Распределение по категориям:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat}: {count}")
    
    # Создаём JSON с метаданными
    metadata = {
        'total_tasks': len(tasks),
        'by_difficulty': {k: len(v) for k, v in by_difficulty.items()},
        'by_category': categories,
        'files_created': [
            'chemistry_all.txt',
            'chemistry_easy.txt',
            'chemistry_medium.txt',
            'chemistry_hard.txt'
        ]
    }
    
    (output_path / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    print("\n✅ Метаданные сохранены в metadata.json")

if __name__ == "__main__":
    # Использование
    create_training_files(
        input_json='data/chemistry_tasks.json',
        output_dir='data/chemistry'
    )
```

#### 1.4. Запуск конвертации

```bash
# Создайте директорию
mkdir -p data/chemistry

# Положите ваш JSON датасет в data/chemistry_tasks.json
# Затем запустите конвертацию:
poetry run python data_preparation/convert_chemistry_data.py
```

**Результат:**
```
data/chemistry/
├── chemistry_all.txt      # Все задачи (для основного обучения)
├── chemistry_easy.txt     # Простые (для быстрого inference)
├── chemistry_medium.txt   # Средние
├── chemistry_hard.txt     # Сложные (возможно, лучше GPT-4)
└── metadata.json          # Статистика датасета
```

---

### Фаза 2: Обучение модели (1-3 дня)

#### 2.1. Выбор конфигурации модели

**Варианты:**

| Конфиг | Параметры | RAM | Время обучения | Качество | Рекомендация |
|--------|-----------|-----|----------------|----------|--------------|
| `small` | ~3M | 2GB | 1-2 часа | 🟡 Среднее | Прототип |
| `medium` | ~19M | 8GB | 4-8 часов | 🟢 Хорошее | ✅ Рекомендуется |
| `base` | ~85M | 16GB+ | 1-2 дня | 🟢🟢 Отличное | Production |

**Рекомендация для химии:** `medium` — оптимальный баланс качество/ресурсы

#### 2.2. Обучение

```bash
cd /Users/Viachaslau_Kazakou/Work/LLM-learn/pytorch_llm

# Вариант 1: Обучение с нуля
poetry run python train.py \
  --data data/chemistry/chemistry_all.txt \
  --config medium \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.0003 \
  --patience 10

# Вариант 2: Fine-tuning существующей модели
poetry run python train.py \
  --data data/chemistry/chemistry_all.txt \
  --config medium \
  --epochs 10 \
  --batch-size 32 \
  --lr 0.0001 \
  --continue-from checkpoints/best_model.pt \
  --patience 5
```

**Параметры обучения:**
- `--epochs 30`: Больше эпох для лучшего запоминания паттернов
- `--batch-size 32`: Зависит от вашей GPU памяти
- `--lr 0.0003`: Learning rate (начальный)
- `--patience 10`: Early stopping после 10 проверок без улучшения

#### 2.3. Мониторинг обучения

Используйте Streamlit UI для мониторинга:

```bash
poetry run streamlit run app.py
```

Или смотрите логи:

```bash
tail -f logs/training.log
```

**Целевые метрики:**
- `train_loss < 1.5` — модель обучается
- `val_loss < 2.0` — хорошее обобщение
- `perplexity < 10` — приемлемое качество

#### 2.4. Оценка качества

Создайте `evaluation/test_chemistry_model.py`:

```python
"""
Тестирование химической модели на контрольных задачах.
"""

from pathlib import Path
from inference import load_model_for_inference, generate_text
from data import CharTokenizer
from config import get_device

# Тестовые задачи
TEST_TASKS = [
    {
        "question": "Закончите уравнение: 2H₂ + O₂ → ?",
        "expected": "2H₂O",
        "category": "уравнения"
    },
    {
        "question": "Молярная масса H₂SO₄?",
        "expected": "98",
        "category": "расчеты"
    },
    {
        "question": "Валентность кислорода?",
        "expected": "II",
        "category": "теория"
    }
]

def test_model(checkpoint_path: str):
    """Тестирует модель на контрольных задачах."""
    
    # Загружаем модель
    device = get_device()
    model, checkpoint = load_model_for_inference(checkpoint_path, device)
    
    # Создаём токенизатор (используйте те же данные, что при обучении!)
    data_path = Path('data/chemistry/chemistry_all.txt')
    text = data_path.read_text(encoding='utf-8')
    tokenizer = CharTokenizer(text)
    
    print(f"🧪 Тестирование модели: {checkpoint_path}")
    print(f"📊 Vocab size: {len(tokenizer.vocab)}\n")
    
    correct = 0
    total = len(TEST_TASKS)
    
    for i, task in enumerate(TEST_TASKS, 1):
        prompt = f"Задача: {task['question']}\nОтвет:"
        
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=50,
            temperature=0.3,  # Низкая для точности
            top_k=20,
            device=device
        )
        
        # Извлекаем ответ
        answer = generated.split("Ответ:")[-1].strip()
        
        # Проверяем
        is_correct = task['expected'].lower() in answer.lower()
        status = "✅" if is_correct else "❌"
        
        print(f"{status} Задача {i} ({task['category']})")
        print(f"   Вопрос: {task['question']}")
        print(f"   Ожидалось: {task['expected']}")
        print(f"   Получено: {answer}")
        print()
        
        if is_correct:
            correct += 1
    
    accuracy = (correct / total) * 100
    print(f"📈 Точность: {correct}/{total} ({accuracy:.1f}%)")
    
    return accuracy

if __name__ == "__main__":
    test_model("checkpoints/best_model.pt")
```

Запуск:

```bash
poetry run python evaluation/test_chemistry_model.py
```

---

### Фаза 3: Создание Router (1 день)

#### 3.1. Реализация классификатора

Создайте `chemistry_router.py`:

```python
"""
Интеллектуальный роутер для AI-репетитора по химии.
Распределяет запросы между локальной моделью и GPT-4.
"""

from openai import OpenAI
import time
from typing import Dict, Literal
from dataclasses import dataclass
import re

# ═══════════════════════════════════════════════════════════════
# Конфигурация
# ═══════════════════════════════════════════════════════════════

@dataclass
class RouterConfig:
    """Конфигурация роутера."""
    local_api_url: str = "http://localhost:8000/v1"
    local_api_key: str = "dummy"
    openai_api_key: str = ""  # Заполните ваш ключ
    
    # Параметры генерации
    local_temperature: float = 0.3  # Низкая для точности
    local_max_tokens: int = 200
    gpt4_temperature: float = 0.7
    gpt4_max_tokens: int = 500
    
    # Стоимость (примерная)
    gpt4_cost_per_1k_tokens: float = 0.03

# ═══════════════════════════════════════════════════════════════
# Классификация сложности
# ═══════════════════════════════════════════════════════════════

DifficultyLevel = Literal["easy", "medium", "hard"]

def classify_question_difficulty(question: str) -> DifficultyLevel:
    """
    Определяет сложность вопроса.
    
    Логика:
    - easy: Типовые задачи (формулы, уравнения, простые расчеты)
    - medium: Задачи средней сложности (многоступенчатые расчеты)
    - hard: Объяснения, механизмы, творческие вопросы
    """
    question_lower = question.lower()
    
    # Паттерны для простых задач (easy) → локальная модель
    easy_patterns = [
        r'закончите уравнение',
        r'уравнение реакции',
        r'молярная масса',
        r'атомная масса',
        r'валентность',
        r'степень окисления',
        r'формула',
        r'число протонов',
        r'число электронов',
        r'периодическая таблица',
        r'группа элементов',
        r'период элемента',
    ]
    
    for pattern in easy_patterns:
        if re.search(pattern, question_lower):
            return "easy"
    
    # Паттерны для сложных вопросов (hard) → GPT-4
    hard_patterns = [
        r'почему',
        r'объясните',
        r'опишите механизм',
        r'как можно объяснить',
        r'в чем причина',
        r'сравните',
        r'предложите эксперимент',
        r'придумайте',
        r'creative',
        r'дискуссия',
    ]
    
    for pattern in hard_patterns:
        if re.search(pattern, question_lower):
            return "hard"
    
    # Длинные вопросы обычно сложные
    if len(question) > 200:
        return "hard"
    
    # Проверка на наличие специфичных терминов (органика, механизмы)
    complex_terms = [
        'механизм',
        'электронные эффекты',
        'резонанс',
        'ароматичность',
        'изомерия',
        'таутомерия',
    ]
    
    if any(term in question_lower for term in complex_terms):
        return "hard"
    
    # По умолчанию — средняя сложность (используем локальную модель)
    return "medium"

# ═══════════════════════════════════════════════════════════════
# Роутер
# ═══════════════════════════════════════════════════════════════

class ChemistryRouter:
    """Роутер для маршрутизации химических вопросов."""
    
    def __init__(self, config: RouterConfig):
        self.config = config
        
        # Локальная модель
        self.local_client = OpenAI(
            base_url=config.local_api_url,
            api_key=config.local_api_key
        )
        
        # GPT-4
        if config.openai_api_key:
            self.gpt4_client = OpenAI(api_key=config.openai_api_key)
        else:
            self.gpt4_client = None
        
        # Метрики
        self.stats = {
            'total_requests': 0,
            'local_requests': 0,
            'gpt4_requests': 0,
            'total_cost': 0.0,
            'total_latency': 0.0,
        }
    
    def answer(self, question: str, force_model: str = None) -> Dict:
        """
        Отвечает на химический вопрос.
        
        Args:
            question: Вопрос ученика
            force_model: Принудительно использовать 'local' или 'gpt4'
        
        Returns:
            {
                'answer': str,
                'model': 'local' | 'gpt4',
                'difficulty': 'easy' | 'medium' | 'hard',
                'cost': float,
                'latency_ms': int
            }
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        # Определяем сложность
        if force_model:
            difficulty = "unknown"
            use_local = (force_model == "local")
        else:
            difficulty = classify_question_difficulty(question)
            use_local = difficulty in ["easy", "medium"]
        
        # Генерируем ответ
        if use_local:
            result = self._answer_local(question)
            model_used = "local"
            cost = 0.0
            self.stats['local_requests'] += 1
        else:
            if not self.gpt4_client:
                # Fallback на локальную модель, если GPT-4 недоступен
                result = self._answer_local(question)
                model_used = "local (fallback)"
                cost = 0.0
                self.stats['local_requests'] += 1
            else:
                result = self._answer_gpt4(question)
                model_used = "gpt4"
                # Приблизительный расчёт стоимости
                tokens_used = len(question) + len(result)
                cost = (tokens_used / 1000) * self.config.gpt4_cost_per_1k_tokens
                self.stats['gpt4_requests'] += 1
                self.stats['total_cost'] += cost
        
        latency_ms = int((time.time() - start_time) * 1000)
        self.stats['total_latency'] += latency_ms
        
        return {
            'answer': result,
            'model': model_used,
            'difficulty': difficulty,
            'cost': cost,
            'latency_ms': latency_ms
        }
    
    def _answer_local(self, question: str) -> str:
        """Отвечает через локальную модель."""
        system_prompt = "Ты — репетитор по химии. Отвечай кратко и точно."
        
        response = self.local_client.chat.completions.create(
            model="custom-llm",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Задача: {question}\nОтвет:"}
            ],
            temperature=self.config.local_temperature,
            max_tokens=self.config.local_max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    def _answer_gpt4(self, question: str) -> str:
        """Отвечает через GPT-4."""
        system_prompt = (
            "Ты — опытный репетитор по химии. "
            "Объясняй понятно, подробно и с примерами. "
            "Используй химические формулы и уравнения."
        )
        
        response = self.gpt4_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=self.config.gpt4_temperature,
            max_tokens=self.config.gpt4_max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    def get_stats(self) -> Dict:
        """Возвращает статистику использования."""
        total = self.stats['total_requests']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'local_percentage': (self.stats['local_requests'] / total) * 100,
            'gpt4_percentage': (self.stats['gpt4_requests'] / total) * 100,
            'avg_latency_ms': self.stats['total_latency'] / total,
            'avg_cost_per_request': self.stats['total_cost'] / total if total > 0 else 0,
        }

# ═══════════════════════════════════════════════════════════════
# Пример использования
# ═══════════════════════════════════════════════════════════════

def main():
    """Демонстрация работы роутера."""
    
    # Конфигурация
    config = RouterConfig(
        openai_api_key="sk-..."  # Замените на ваш ключ
    )
    
    router = ChemistryRouter(config)
    
    # Тестовые вопросы
    test_questions = [
        "Закончите уравнение: 2H₂ + O₂ → ?",
        "Молярная масса H₂SO₄?",
        "Почему вода — хороший растворитель?",
        "Объясните механизм электрофильного присоединения к алкену",
        "Валентность углерода в метане?",
    ]
    
    print("🧪 Тестирование Chemistry Router\n")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Вопрос {i}: {question}")
        
        result = router.answer(question)
        
        print(f"🤖 Модель: {result['model']}")
        print(f"📊 Сложность: {result['difficulty']}")
        print(f"💰 Стоимость: ${result['cost']:.4f}")
        print(f"⏱️  Время: {result['latency_ms']}ms")
        print(f"💬 Ответ: {result['answer'][:200]}...")
        print("-" * 60)
    
    # Статистика
    stats = router.get_stats()
    print("\n📈 Статистика:")
    print(f"   Всего запросов: {stats['total_requests']}")
    print(f"   Локальная модель: {stats['local_requests']} ({stats['local_percentage']:.1f}%)")
    print(f"   GPT-4: {stats['gpt4_requests']} ({stats['gpt4_percentage']:.1f}%)")
    print(f"   Общая стоимость: ${stats['total_cost']:.2f}")
    print(f"   Средняя стоимость: ${stats['avg_cost_per_request']:.4f}")
    print(f"   Среднее время: {stats['avg_latency_ms']:.0f}ms")

if __name__ == "__main__":
    main()
```

#### 3.2. Тестирование роутера

```bash
# Сначала запустите локальную модель
poetry run uvicorn api_server:app --host 0.0.0.0 --port 8000 &

# В другом терминале запустите роутер
poetry run python chemistry_router.py
```

**Ожидаемый результат:**
```
🧪 Тестирование Chemistry Router

📝 Вопрос 1: Закончите уравнение: 2H₂ + O₂ → ?
🤖 Модель: local
📊 Сложность: easy
💰 Стоимость: $0.0000
⏱️  Время: 45ms
💬 Ответ: 2H₂O

📝 Вопрос 3: Почему вода — хороший растворитель?
🤖 Модель: gpt4
📊 Сложность: hard
💰 Стоимость: $0.0024
⏱️  Время: 1850ms
💬 Ответ: Вода является полярным растворителем благодаря...

📈 Статистика:
   Всего запросов: 5
   Локальная модель: 3 (60.0%)
   GPT-4: 2 (40.0%)
   Общая стоимость: $0.05
   Средняя стоимость: $0.0100
   Среднее время: 520ms
```

---

### Фаза 4: Интеграция и развертывание (1-2 дня)

#### 4.1. Backend API

Создайте `app_chemistry.py`:

```python
"""
Backend API для AI-репетитора по химии.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
from datetime import datetime

from chemistry_router import ChemistryRouter, RouterConfig

# ═══════════════════════════════════════════════════════════════
# Модели данных
# ═══════════════════════════════════════════════════════════════

class QuestionRequest(BaseModel):
    question: str
    force_model: Optional[str] = None  # 'local' или 'gpt4'
    user_id: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    model: str
    difficulty: str
    cost: float
    latency_ms: int
    timestamp: str

class StatsResponse(BaseModel):
    total_requests: int
    local_requests: int
    gpt4_requests: int
    local_percentage: float
    gpt4_percentage: float
    total_cost: float
    avg_cost_per_request: float
    avg_latency_ms: float

# ═══════════════════════════════════════════════════════════════
# Инициализация
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Chemistry Tutor API",
    description="Hybrid AI репетитор по химии",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Роутер
config = RouterConfig(
    openai_api_key="sk-..."  # Замените на ваш ключ!
)
router = ChemistryRouter(config)

# ═══════════════════════════════════════════════════════════════
# Эндпоинты
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Информация об API."""
    return {
        "name": "Chemistry Tutor API",
        "version": "1.0.0",
        "description": "Hybrid AI репетитор по химии",
        "endpoints": {
            "ask": "POST /api/ask",
            "stats": "GET /api/stats"
        }
    }

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Задать вопрос репетитору.
    
    Пример запроса:
        {
          "question": "Молярная масса H₂SO₄?",
          "force_model": null,
          "user_id": "student_123"
        }
    """
    try:
        result = router.answer(
            question=request.question,
            force_model=request.force_model
        )
        
        return AnswerResponse(
            **result,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Получить статистику использования."""
    stats = router.get_stats()
    return StatsResponse(**stats)

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# ═══════════════════════════════════════════════════════════════
# Запуск
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

#### 4.2. Запуск системы

**Terminal 1:** Локальная модель
```bash
cd /Users/Viachaslau_Kazakou/Work/LLM-learn/pytorch_llm
poetry run uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**Terminal 2:** Backend API
```bash
poetry run uvicorn app_chemistry:app --host 0.0.0.0 --port 8001
```

**Terminal 3:** Тест
```bash
curl http://localhost:8001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Молярная масса H₂SO₄?"}'
```

#### 4.3. Frontend интеграция (пример React)

```typescript
// api/chemistry.ts
export interface QuestionRequest {
  question: string;
  force_model?: 'local' | 'gpt4';
  user_id?: string;
}

export interface AnswerResponse {
  answer: string;
  model: string;
  difficulty: string;
  cost: number;
  latency_ms: number;
  timestamp: string;
}

export async function askQuestion(request: QuestionRequest): Promise<AnswerResponse> {
  const response = await fetch('http://localhost:8001/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    throw new Error('Failed to get answer');
  }
  
  return response.json();
}

// components/ChemistryChat.tsx
import React, { useState } from 'react';
import { askQuestion } from '../api/chemistry';

export function ChemistryChat() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelUsed, setModelUsed] = useState<string>('');
  const [cost, setCost] = useState<number>(0);

  const handleAsk = async () => {
    setLoading(true);
    try {
      const result = await askQuestion({ question });
      setAnswer(result.answer);
      setModelUsed(result.model);
      setCost(result.cost);
    } catch (error) {
      console.error(error);
      alert('Ошибка при получении ответа');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chemistry-chat">
      <h2>AI Репетитор по химии</h2>
      
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Задайте вопрос по химии..."
        rows={4}
      />
      
      <button onClick={handleAsk} disabled={loading}>
        {loading ? 'Думаю...' : 'Спросить'}
      </button>
      
      {answer && (
        <div className="answer">
          <h3>Ответ:</h3>
          <p>{answer}</p>
          <div className="meta">
            <span>Модель: {modelUsed}</span>
            <span>Стоимость: ${cost.toFixed(4)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
```

---

### Фаза 5: Метрики и оптимизация (постоянно)

#### 5.1. Мониторинг метрик

Создайте `monitoring/metrics_dashboard.py`:

```python
"""
Dashboard для мониторинга метрик Chemistry Router.
"""

import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Chemistry Tutor Metrics", layout="wide")

# ═══════════════════════════════════════════════════════════════
# Получение данных
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def get_stats():
    """Получает статистику с API."""
    response = requests.get("http://localhost:8001/api/stats")
    return response.json()

# ═══════════════════════════════════════════════════════════════
# Dashboard
# ═══════════════════════════════════════════════════════════════

st.title("🧪 Chemistry Tutor — Метрики")

# Получаем данные
stats = get_stats()

# Основные метрики
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Всего запросов",
        stats['total_requests']
    )

with col2:
    st.metric(
        "Локальная модель",
        f"{stats['local_percentage']:.1f}%",
        f"{stats['local_requests']} запросов"
    )

with col3:
    st.metric(
        "Стоимость",
        f"${stats['total_cost']:.2f}",
        f"${stats['avg_cost_per_request']:.4f}/запрос"
    )

with col4:
    st.metric(
        "Среднее время",
        f"{stats['avg_latency_ms']:.0f}ms"
    )

# График распределения
st.subheader("📊 Распределение запросов")

fig = px.pie(
    values=[stats['local_requests'], stats['gpt4_requests']],
    names=['Локальная модель (бесплатно)', 'GPT-4 (платно)'],
    title="Какая модель используется чаще?",
    color_discrete_sequence=['#00d084', '#ff6b6b']
)
st.plotly_chart(fig, use_container_width=True)

# Прогноз экономии
st.subheader("💰 Прогноз экономии")

col1, col2 = st.columns(2)

with col1:
    requests_per_month = st.number_input(
        "Запросов в месяц",
        min_value=1000,
        max_value=1000000,
        value=30000,
        step=1000
    )

with col2:
    gpt4_cost_per_1k = st.number_input(
        "Стоимость GPT-4 (за 1k токенов)",
        min_value=0.01,
        max_value=0.10,
        value=0.03,
        step=0.01
    )

# Расчёт
only_gpt4_cost = (requests_per_month / 1000) * gpt4_cost_per_1k * 10  # ~10 токенов средний запрос
hybrid_cost = only_gpt4_cost * (stats['gpt4_percentage'] / 100)
savings = only_gpt4_cost - hybrid_cost
savings_yearly = savings * 12

st.markdown(f"""
### Результаты:

| Сценарий | Стоимость/месяц | Стоимость/год |
|----------|----------------|---------------|
| **Только GPT-4** | ${only_gpt4_cost:.2f} | ${only_gpt4_cost * 12:.2f} |
| **Hybrid (текущий)** | ${hybrid_cost:.2f} | ${hybrid_cost * 12:.2f} |
| **💰 Экономия** | **${savings:.2f}** | **${savings_yearly:.2f}** |

**Процент экономии:** {(savings / only_gpt4_cost * 100):.1f}%
""")

# Обновление
st.button("🔄 Обновить данные")
```

Запуск:
```bash
poetry run streamlit run monitoring/metrics_dashboard.py
```

#### 5.2. A/B тестирование качества

Создайте тесты для сравнения качества ответов:

```python
# evaluation/ab_testing.py
"""
A/B тестирование: локальная модель vs GPT-4.
"""

from chemistry_router import ChemistryRouter, RouterConfig
import json

def ab_test(test_questions_file: str = 'evaluation/test_questions.json'):
    """Сравнивает качество ответов."""
    
    config = RouterConfig(openai_api_key="sk-...")
    router = ChemistryRouter(config)
    
    # Загружаем тестовые вопросы
    with open(test_questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    results = []
    
    for q in questions:
        # Локальная модель
        local_result = router.answer(q['question'], force_model='local')
        
        # GPT-4
        gpt4_result = router.answer(q['question'], force_model='gpt4')
        
        results.append({
            'question': q['question'],
            'expected': q['expected_answer'],
            'local_answer': local_result['answer'],
            'gpt4_answer': gpt4_result['answer'],
            'local_latency': local_result['latency_ms'],
            'gpt4_latency': gpt4_result['latency_ms'],
            'cost_diff': gpt4_result['cost'],
        })
    
    # Сохраняем результаты
    with open('evaluation/ab_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✅ A/B тест завершён. Результаты в evaluation/ab_test_results.json")

if __name__ == "__main__":
    ab_test()
```

#### 5.3. Оптимизация классификатора

После сбора данных оптимизируйте правила классификации:

```python
# Анализируйте логи, чтобы найти:
# 1. Ложные срабатывания (simple → отправлено на GPT-4 зря)
# 2. Пропуски (complex → локальная модель не справилась)

# Добавляйте/корректируйте паттерны в classify_question_difficulty()
```

---

## 🚀 Альтернативные подходы

### Подход 1: Fine-tune LLaMA-2 7B (высокое качество)

**Преимущества:**
- 🟢🟢 Отличное качество (близко к GPT-3.5)
- 🟢 Всё равно дешевле GPT-4
- 🟢 Больше параметров = лучше понимание

**Недостатки:**
- 🔴 Сложнее настройка
- 🔴 Требуется GPU с 16+ GB VRAM
- 🔴 Дольше инференс (500-1000ms vs 50ms)

**Когда использовать:**
- Критично высокое качество
- Есть мощный сервер
- Готовы потратить время на настройку

### Подход 2: Retrieval-Augmented Generation (RAG)

**Идея:** Добавить базу знаний (учебники, задачники) + векторный поиск

```python
# Pseudo-code
def answer_with_rag(question):
    # 1. Найти похожие примеры в базе знаний
    similar_examples = vector_search(question, top_k=3)
    
    # 2. Добавить их в контекст
    context = "\n".join([ex['solution'] for ex in similar_examples])
    prompt = f"Примеры:\n{context}\n\nВопрос: {question}\nОтвет:"
    
    # 3. Генерировать ответ
    return local_model.generate(prompt)
```

**Преимущества:**
- 🟢 Модель "видит" похожие примеры
- 🟢 Выше точность на типовых задачах
- 🟢 Не требует больших моделей

**Недостатки:**
- 🟡 Нужна векторная база данных (Pinecone, Weaviate, FAISS)
- 🟡 Дольше разработка

### Подход 3: Ensemble (комбинация моделей)

**Идея:** Используйте несколько моделей и выбирайте лучший ответ

```python
def ensemble_answer(question):
    # Генерируем 3 ответа разными моделями
    answers = [
        local_model.generate(question),
        another_local_model.generate(question),
        gpt4.generate(question)
    ]
    
    # Выбираем лучший (по confidence или голосованию)
    best_answer = select_best(answers)
    return best_answer
```

---

## 📊 Ожидаемые результаты

### Метрики успеха:

| Метрика | До (только GPT-4) | После (Hybrid) | Улучшение |
|---------|-------------------|----------------|-----------|
| **Стоимость/месяц** | $60 | $18 | 💰 -70% |
| **Latency (простые)** | 1500ms | 50ms | ⚡ -97% (30x быстрее) |
| **Latency (сложные)** | 1500ms | 1500ms | = |
| **Точность (простые)** | 95% | 90-95% | ≈ |
| **Точность (сложные)** | 95% | 95% | = |
| **Приватность** | ❌ | ✅ 70% локально | ✅ |

### Сценарии использования:

**70% запросов (локальная модель):**
- ✅ Уравнения реакций
- ✅ Молярные массы
- ✅ Формулы
- ✅ Валентности
- ✅ Простые расчёты

**30% запросов (GPT-4):**
- 💡 Объяснения механизмов
- 💡 Творческие задачи
- 💡 Дискуссии
- 💡 Сложные многоступенчатые задачи

---

## ⚠️ Важные замечания и ограничения

### 1. Качество локальной модели зависит от данных

**Правило:** Качество OUT = Качество IN

```
1,000 задач   → 🟡 Среднее качество (60-70% точность)
5,000 задач   → 🟢 Хорошее качество (80-90% точность)
20,000 задач  → 🟢🟢 Отличное качество (90-95% точность)
```

### 2. Размер модели имеет значение

| Конфиг | Параметры | Качество простых задач | Качество сложных задач |
|--------|-----------|----------------------|----------------------|
| `small` | 3M | 🟡 60-70% | 🔴 30-40% |
| `medium` | 19M | 🟢 80-90% | 🟡 50-60% |
| `base` | 85M | 🟢 85-95% | 🟢 70-80% |

**Рекомендация:** Для production используйте `medium` или `base`

### 3. Токенизатор CharTokenizer — ограничение

**Проблемы:**
- Длинные последовательности (каждая буква = токен)
- Не понимает семантику слов
- Большой context требует больше памяти

**Решение для v2:**
- Переобучите модель с tiktoken или SentencePiece
- Используйте BPE токенизацию

### 4. Мониторинг и доработка

**Постоянно отслеживайте:**
- Ложные срабатывания (simple → GPT-4 зря)
- Ошибки локальной модели (где она слабая?)
- Фидбек учеников (thumbs up/down)

**Итеративно улучшайте:**
- Добавляйте новые данные в датасет
- Корректируйте правила классификации
- Fine-tune модель на ошибках

---

## 🎯 Roadmap развития

### Версия 1.0 (MVP) — 1-2 недели
- [x] Базовая модель обучена
- [ ] Router реализован
- [ ] Backend API работает
- [ ] Основная интеграция
- [ ] Датасет: 1,000+ задач
- [ ] Конфиг: `small` или `medium`

**Цель:** Доказать концепцию, начать экономить

### Версия 1.1 (Улучшения) — 1 месяц
- [ ] Больше данных (5,000+ задач)
- [ ] Модель `medium` или `base`
- [ ] Мониторинг метрик
- [ ] A/B тестирование
- [ ] Оптимизация классификатора

**Цель:** Повысить качество до 85-90%

### Версия 2.0 (Advanced) — 2-3 месяца
- [ ] Переход на tiktoken (BPE)
- [ ] RAG (векторный поиск по задачникам)
- [ ] Кеширование популярных вопросов
- [ ] Fine-tune LLaMA-2 7B (опционально)
- [ ] Мультимодальность (распознавание формул с фото)

**Цель:** Production-ready система

### Версия 3.0 (Scale) — 6+ месяцев
- [ ] Автоматическое обновление датасета
- [ ] Персонализация (модель адаптируется под ученика)
- [ ] Микросервисная архитектура
- [ ] Kubernetes deployment
- [ ] Мониторинг (Prometheus, Grafana)

**Цель:** Масштабирование на тысячи учеников

---

## 📚 Ресурсы и документация

### Созданные файлы:

1. `api_server.py` — OpenAI-compatible API для локальной модели
2. `chemistry_router.py` — Интеллектуальный роутер
3. `app_chemistry.py` — Backend API
4. `data_preparation/convert_chemistry_data.py` — Конвертация данных
5. `evaluation/test_chemistry_model.py` — Тестирование модели
6. `evaluation/ab_testing.py` — A/B тесты
7. `monitoring/metrics_dashboard.py` — Dashboard метрик

### Документация:

- `docs/DEPLOYMENT.md` — Гайд по deployment
- `docs/QUICKSTART_API.md` — Быстрый старт API
- `docs/HYBRID_AI_TUTOR_PLAN.md` — Этот документ

### Внешние ресурсы:

- PyTorch: https://pytorch.org/docs
- FastAPI: https://fastapi.tiangolo.com
- OpenAI API: https://platform.openai.com/docs
- tiktoken: https://github.com/openai/tiktoken
- Streamlit: https://docs.streamlit.io

---

## ✅ Чеклист внедрения

### Подготовка:
- [ ] Собрать датасет химических задач (минимум 1,000)
- [ ] Конвертировать в формат для обучения
- [ ] Установить зависимости (`poetry add openai`)
- [ ] Получить OpenAI API ключ

### Обучение:
- [ ] Обучить модель на химических данных
- [ ] Протестировать на контрольных задачах
- [ ] Достичь accuracy > 80% на простых задачах
- [ ] Сохранить best_model.pt

### Интеграция:
- [ ] Реализовать chemistry_router.py
- [ ] Протестировать классификацию
- [ ] Создать Backend API (app_chemistry.py)
- [ ] Интегрировать с фронтендом

### Развертывание:
- [ ] Запустить api_server (локальная модель) на production
- [ ] Запустить app_chemistry (backend) на production
- [ ] Настроить мониторинг и логирование
- [ ] Протестировать end-to-end

### Оптимизация:
- [ ] Собирать метрики (latency, cost, accuracy)
- [ ] Анализировать ошибки
- [ ] Корректировать правила классификации
- [ ] Дообучать модель на новых данных

---

## 💡 Заключение

**Hybrid AI подход** — оптимальное решение для AI-репетитора:

✅ **Экономия:** 70% расходов  
✅ **Скорость:** 30x быстрее для простых задач  
✅ **Качество:** Сохраняется для сложных случаев  
✅ **Приватность:** 70% запросов обрабатываются локально  
✅ **Масштабируемость:** Легко добавлять новые модели  

**Следующий шаг:** Соберите датасет и начните обучение! 🚀

---

**Автор:** AI Assistant  
**Дата:** 30 мая 2026  
**Версия:** 1.0
