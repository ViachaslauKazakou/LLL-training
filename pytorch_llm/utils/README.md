# Утилиты для работы с данными

Набор инструментов для подготовки датасетов для обучения LLM.

## 📄 pdf_to_text.py — Конвертер PDF в текст

Конвертирует PDF файлы в чистый текст, пригодный для обучения.

### Возможности:
- ✅ Извлечение текста из PDF
- ✅ Автоматическая очистка (убирает лишние пробелы, форматирование)
- ✅ Пакетная обработка целых директорий
- ✅ Поддержка паттернов для фильтрации файлов

### Использование:

**Один файл:**
```bash
cd pytorch_llm/utils
python pdf_to_text.py book.pdf output.txt
```

**Вся директория:**
```bash
python pdf_to_text.py --dir pdfs/ --output-dir texts/
```

**С паттерном (только химия):**
```bash
python pdf_to_text.py --dir pdfs/ --output-dir texts/ --pattern "chemistry_*.pdf"
```

**Без очистки:**
```bash
python pdf_to_text.py book.pdf output.txt --no-clean
```

### Пример вывода:
```
📄 Обработка: chemistry_book.pdf
  Страница 150/150...
  ✅ Извлечено 245,389 символов из 150 страниц
  💾 Сохранено в: texts/chemistry_book.txt

====================================================================
✅ Конвертировано: 3/3 файлов
📊 Всего символов: 1,245,389
📊 Примерный размер: 1217.2 KB
====================================================================
```

---

## 🔗 merge_datasets.py — Объединитель датасетов

Объединяет несколько текстовых файлов в один и анализирует содержимое.

### Возможности:
- ✅ Объединение нескольких файлов
- ✅ Автоматический анализ (язык, размер, статистика)
- ✅ Добавление заголовков источников
- ✅ Настраиваемые разделители
- ✅ Поддержка паттернов

### Использование:

**Объединить конкретные файлы:**
```bash
cd pytorch_llm/utils
python merge_datasets.py file1.txt file2.txt file3.txt --output combined.txt
```

**Объединить всю директорию:**
```bash
python merge_datasets.py --dir ../data/ --output combined_dataset.txt
```

**Только химические файлы:**
```bash
python merge_datasets.py --dir ../data/ --pattern "chemistry_*.txt" --output chemistry_dataset.txt
```

**Анализ датасета:**
```bash
python merge_datasets.py --analyze combined_dataset.txt
```

**Без заголовков:**
```bash
python merge_datasets.py file1.txt file2.txt --output out.txt --no-headers
```

### Пример вывода:

**При объединении:**
```
📚 Объединение 3 файлов:

============================================================
📊 АНАЛИЗ ИСХОДНЫХ ФАЙЛОВ:
============================================================
  1. chemistry_textbook.txt                    1217.2 KB,  25,340 строк
  2. chemistry_wikipedia.txt                    423.5 KB,   8,920 строк
  3. chemistry_school_belarus.txt                44.0 KB,     899 строк

  Итого:                                       1684.7 KB
============================================================

✅ Объединенный файл сохранен: combined_dataset.txt

============================================================
📊 АНАЛИЗ ДАТАСЕТА: combined_dataset.txt
============================================================

📏 Размер:
  • Байт:     1,725,440
  • KB:       1684.81
  • MB:       1.64

📝 Контент:
  • Символов: 1,725,440
  • Слов:     245,678
  • Строк:    35,159
  • Пустых:   2,345

🌐 Язык:
  • Кириллица:
    - Символов: 1,650,230
    - Строк:    33,450 (95.1%)
  • Латиница:
    - Символов: 45,678
    - Строк:    1,234 (3.5%)

============================================================
```

---

## 🚀 Быстрый старт: Подготовка датасета по химии

### Шаг 1: Конвертируем PDF
```bash
cd pytorch_llm/utils

# Конвертируем все PDF из папки
python pdf_to_text.py --dir ~/Downloads/chemistry_pdfs/ --output-dir ../data/chemistry_texts/
```

### Шаг 2: Объединяем с существующими данными
```bash
# Объединяем новые тексты с имеющимися
python merge_datasets.py \
    --dir ../data/ \
    --pattern "chemistry_*.txt" \
    --output ../data/chemistry_combined.txt
```

### Шаг 3: Анализируем результат
```bash
# Проверяем что получилось
python merge_datasets.py --analyze ../data/chemistry_combined.txt
```

### Шаг 4: Обучаем модель
```bash
cd ../..
make pytorch

# В UI:
# Dataset: pytorch_llm/data/chemistry_combined.txt
# Tokenizer: TikToken (cl100k_base)
# Model: small
```

---

## 📋 Чеклист подготовки датасета

- [ ] Собрать PDF файлы (книги, учебники, статьи)
- [ ] Конвертировать PDF → текст
- [ ] Проверить качество конвертации (открыть в редакторе)
- [ ] Объединить с существующими данными
- [ ] Проанализировать итоговый датасет
- [ ] Проверить язык (кириллица/латиница)
- [ ] Убедиться что размер > 5 MB для русского, > 10 MB для английского
- [ ] Запустить обучение

---

## 💡 Советы

**Качество PDF:**
- Используйте PDF с текстовым слоем (не сканы!)
- Избегайте PDF с таблицами и формулами
- Лучше несколько маленьких качественных PDF, чем один большой плохой

**Размер датасета:**
- Минимум 5 MB для русского текста
- Минимум 10 MB для английского
- Оптимально 20-50 MB для хорошего качества

**Очистка:**
- Используйте `--no-clean` только если текст уже чистый
- После конвертации проверьте результат в редакторе
- Удалите служебную информацию (оглавления, номера страниц) вручную

**Язык:**
- Используйте `--analyze` для проверки языка
- Если < 90% нужного языка — пересмотрите источники
- Смешение языков (русский + английский) ухудшает качество

---

## ⚙️ Требования

```bash
# Установка зависимостей
poetry add pypdf

# Или через pip
pip install pypdf
```

---

## 🔧 Опции

### pdf_to_text.py
```
--dir DIR              Директория с PDF
--output-dir DIR       Директория для текста
--pattern PATTERN      Паттерн поиска (default: *.pdf)
--no-clean            Не очищать текст
```

### merge_datasets.py
```
--dir DIR              Директория с текстами
--pattern PATTERN      Паттерн поиска (default: *.txt)
--output FILE         Выходной файл
--separator SEP       Разделитель (default: \n\n---\n\n)
--no-headers          Не добавлять заголовки
--analyze FILE        Анализировать файл
--no-analyze          Не показывать анализ
```

---

## 📝 Примеры workflow

### Workflow 1: Химия из учебников
```bash
# 1. Конвертируем PDF учебники
python pdf_to_text.py --dir ~/chemistry_books/ --output-dir ../data/chem_texts/

# 2. Объединяем с имеющимися данными
python merge_datasets.py \
    ../data/chemistry_school_belarus.txt \
    ../data/chem_texts/*.txt \
    --output ../data/chemistry_full.txt

# 3. Анализируем
python merge_datasets.py --analyze ../data/chemistry_full.txt

# 4. Обучаем
cd ../..
poetry run streamlit run pytorch_llm/app.py
```

### Workflow 2: Python документация
```bash
# 1. Конвертируем PDF книги по Python
python pdf_to_text.py --dir ~/python_books/ --output-dir ../data/python_texts/

# 2. Объединяем (только русские)
python merge_datasets.py \
    --dir ../data/ \
    --pattern "python_*_ru.txt" \
    --output ../data/python_russian_full.txt

# 3. Проверяем язык
python merge_datasets.py --analyze ../data/python_russian_full.txt
```

---

## 🐛 Известные проблемы

**PDF со сканами:**
- Утилита не распознает текст на изображениях
- Используйте OCR (Tesseract) отдельно

**Формулы:**
- Математические формулы могут некорректно конвертироваться
- Для химии: формулы типа H₂O, CO₂ работают нормально

**Кодировка:**
- Все файлы сохраняются в UTF-8
- Поддержка кириллицы и латиницы

---

## 📚 Дополнительные ресурсы

- [pypdf документация](https://pypdf.readthedocs.io/)
- [Лучшие практики датасетов для LLM](../docs/dataset_best_practices.md)
- [Troubleshooting обучения](../docs/troubleshooting.md)
