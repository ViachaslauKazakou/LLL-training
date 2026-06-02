# 📷 OCR — Распознавание текста из сканов

Отдельный модуль для распознавания текста из изображений учебников.

---

## 🎯 Назначение

**Разделение ответственности (SOLID):**
- `ocr/` — **только распознавание** и подготовка данных
- `pytorch_llm/` — **только обучение** языковых моделей

---

## 📁 Структура

```
ocr/
├── ocr_app.py              # Streamlit UI для распознавания
├── data_preparation/       # Временные файлы OCR
│   ├── scans/              # Исходные изображения (загружаются пользователем)
│   └── tasks/              # Распознанные задачи (JSON)
└── README.md               # Эта документация
```

---

## 🚀 Использование

### **1. Запуск OCR приложения:**

```bash
make tesseract
# Или:
make ocr
```

Откроется: **http://localhost:8503**

---

### **2. Workflow распознавания:**

```
1. Загрузка сканов
   └─ Tab 1: Upload JPEG/PNG → data_preparation/scans/

2. Распознавание текста
   └─ Tab 2: Tesseract OCR → редактирование WYSIWYG

3. Редактирование задач
   └─ Tab 3: Правка вопросов/ответов

4. Экспорт в JSON
   └─ Tab 4: Сохранение → data_preparation/tasks/
      Кнопка: "Скопировать в data/" → ../pytorch_llm/data/

5. Объединение датасетов
   └─ Tab 5: Merge JSON → data_preparation/tasks/merged_dataset.json
```

---

## 📤 Экспорт данных для обучения

**Из OCR в LLM:**

```bash
# Ручное копирование:
cp ocr/data_preparation/tasks/chemistry_tasks.json pytorch_llm/data/

# Или через UI:
Tab 4 → кнопка "Скопировать в data/"
```

**Результат:**
```
pytorch_llm/data/chemistry_tasks.json  ← готово для обучения
```

---

## 🔧 Технологии

- **Tesseract OCR** — распознавание текста (/opt/homebrew/bin/tesseract)
- **Streamlit** — веб-интерфейс (порт 8503)
- **Pillow** — обработка изображений
- **JavaScript WYSIWYG** — редактор формул (H₂O, CO₂, надстрочные/подстрочные)

---

## ⚙️ Зависимости

```bash
# Tesseract (macOS):
brew install tesseract tesseract-lang

# Python пакеты (уже в pyproject.toml):
poetry install
```

---

## 📝 Формат данных

**Выходной JSON (tasks format):**

```json
{
  "tasks": [
    {
      "question": "Какова молярная масса H₂O?",
      "solution": "M(H₂O) = 2×1 + 16 = 18 г/моль",
      "answer": "18 г/моль"
    }
  ]
}
```

**Конвертация в TXT для LLM:**

```bash
# Автоматически при загрузке в pytorch_llm/data.py
# Или вручную:
cd pytorch_llm
poetry run python -c "from data import convert_tasks_json_to_text; ..."
```

---

## 🎨 Возможности редактора

**WYSIWYG инструменты:**
- **Надстрочный индекс:** 10²³, Nₐ
- **Подстрочный индекс:** H₂O, CO₂
- **Готовые шаблоны:** H₂O, CO₂, H₂SO₄, °C
- **Спецсимволы:** · → ⇌ ± ≈ ≠ ≤ ≥ Δ ∑ √ π

**Как использовать:**
1. Выделите текст
2. Нажмите кнопку "Надстрочный" или "Подстрочный"
3. Текст автоматически конвертируется (H2O → H₂O)

---

## 🔗 Связь с pytorch_llm

**OCR → LLM pipeline:**

```
1. ocr/data_preparation/scans/page1.jpg
   ↓ (Tesseract)
2. ocr/data_preparation/tasks/chemistry_ch1.json
   ↓ (Copy to data/)
3. pytorch_llm/data/chemistry_ch1.json
   ↓ (Training)
4. pytorch_llm/checkpoints/chemistry_model_best.pt
```

**Важно:** OCR **не зависит** от pytorch_llm!  
Можно использовать отдельно для подготовки данных.

---

## 🧹 Очистка

```bash
# Удалить временные файлы:
rm -rf ocr/data_preparation/scans/*
rm -rf ocr/data_preparation/tasks/*

# Оставить только структуру:
mkdir -p ocr/data_preparation/scans
mkdir -p ocr/data_preparation/tasks
```

---

## 📚 Документация

**См. также:**
- [pytorch_llm/README.md](../pytorch_llm/README.md) — обучение моделей
- [docs/JSON_FORMATS.md](../docs/JSON_FORMATS.md) — форматы данных
- [Makefile](../Makefile) — все команды

---

**Автор:** Senior Python ML Engineer  
**Дата:** 2026-06-01  
**Принцип:** Single Responsibility (SOLID)
