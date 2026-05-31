# Mini LLM v3 — Трансформер с нуля на NumPy

Учебная реализация авторегрессионного трансформера (GPT-подобная архитектура) на чистом NumPy — без PyTorch/TensorFlow.

## Архитектура

```
Token Embeddings (E)  → (T, D)
     ↓
+ Positional Encoding → (T, D)
     ↓
Masked Self-Attention → (T, D)
  Q = X @ Wq
  K = X @ Wk
  V = X @ Wv
  A = softmax((Q @ K.T / √D) + causal_mask)
  out = A @ V
     ↓
Take last token [-1]  → (D,)
     ↓
Linear projection (Wo) → (V,)
     ↓
Softmax → probabilities
```

**Causal mask** — треугольная маска, запрещающая токенам видеть будущее (как в GPT).

**Градиентный спуск** — полный backward pass вручную, без autograd.

## Структура

```
mini_llm/v3/
├── corpus.json        # Данные (список предложений)
├── model.py           # Класс MiniLLM — вся архитектура + обучение
├── cli.py             # Интерактивный терминал
├── app.py             # Streamlit UI
└── saved/
    └── model.npz      # Сохранённая модель (веса + словарь + история)
```

## Запуск

### Установка

```bash
make install
```

### Подготовка собственного корпуса

Если у вас есть текстовый файл с предложениями (сообщения с форума, чата, блога):

**Вариант 1: CLI**
```bash
make cli
# → [P] Подготовить корпус из текста
# Укажите путь к .txt файлу → создастся corpus.json
```

**Вариант 2: Streamlit**
```bash
make ui
# → Боковая панель → 📝 Подготовка корпуса
# → Загрузите .txt файл → "🔨 Создать корпус"
```

**Вариант 3: Скрипт**
```bash
cd mini_llm/v3
poetry run python prepare_corpus.py input.txt corpus_output.json \
  --min-words 2 --max-words 15
```

**Формат входного файла:** одна фраза на строку или предложения через точку/перенос строки.

### Консольный интерфейс

```bash
make cli
```

Меню:
- **[P]** Подготовить корпус из текста (загружаете .txt → создаётся corpus.json)
- **[1]** Загрузить корпус
- **[2]** Обучить модель (300–800 эпох, LR=0.01)
- **[3]** Генерировать текст
- **[4]** Предсказать следующее слово (топ-k)
- **[5]** Матрица внимания (ASCII art)
- **[6]** Сохранить модель
- **[7]** Загрузить модель
- **[8]** Информация о модели

### Streamlit UI

```bash
make ui
```

Открывается в браузере:
- **Обучение** — с live loss chart
- **Генерация** — с подсветкой новых слов
- **Предсказание** — топ-k с progress bars
- **Внимание** — heatmap визуализация весов attention

### Быстрое обучение

```bash
make train    # 500 эпох, сохранение в saved/model.npz
```

## Параметры генерации

| Параметр | Что делает |
|----------|------------|
| **Temperature** | `< 1` детерминированный, `> 1` случайный |
| **Top-p (nucleus)** | Выбирает из топ-p% вероятностной массы (0.85 = только 85% самых вероятных слов) |
| **Repetition penalty** | Штраф за повторение уже встречавшихся слов (1.3 = хорошо против зацикливания) |

## Почему результаты не идеальные?

Это **учебная** модель — цель показать архитектуру, а не генерировать осмысленный текст. Ограничения:

| Параметр | Mini LLM | GPT-2 small | GPT-3 |
|----------|----------|-------------|-------|
| `d_model` | 32 | 768 | 12288 |
| `context_len` | 6 | 1024 | 2048 |
| Параметры | ~3K | 117M | 175B |
| Attention heads | 1 | 12 | 96 |

**Нет в этой реализации:**
- Multi-head attention (только 1 голова)
- Feed-forward network (MLP после attention)
- Layer normalization
- Dropout / regularization
- Большие датасеты (сотни тысяч предложений)

## Что реализовано

✅ **Полный forward pass** — embeddings → positional encoding → masked self-attention → projection  
✅ **Полный backward pass** — ручное дифференцирование без autograd  
✅ **SGD с gradient clipping** — стабилизация обучения  
✅ **Nucleus sampling (top-p)** — более разнообразная генерация  
✅ **Repetition penalty** — против зацикливания  
✅ **Сохранение/загрузка** — `.npz` формат (веса + метаданные)  
✅ **Визуализация attention** — heatmap весов внимания  

## Примеры

### Обучение

```python
from model import MiniLLM

llm = MiniLLM(d_model=32, context_len=6)
llm.load_corpus('corpus.json')
llm.train(n_epochs=500, lr=0.01)
llm.save('saved/model.npz')
```

### Генерация

```python
llm.generate(
    'кот сидел', 
    n_new=6, 
    temperature=0.6, 
    top_p=0.85,
    repetition_penalty=1.3
)
# → 'кот сидел радости мурлыкал села от пела'
```

### Предсказание

```python
llm.predict_next(['кот', 'сидел', 'на', 'коврике'], k=5)
# → [('в', 0.439), ('на', 0.212), ('кот', 0.103), ...]
```

### Матрица внимания

```python
weights, tokens = llm.attention_weights(['кот', 'сидел', 'на', 'коврике'])
# weights: (T, T) — веса внимания
# tokens:  список токенов
```

## Принципы кода

- **Clarity over cleverness** — код читается 10× больше, чем пишется
- **Explicit > implicit** — всё явно, никакой магии
- **Progressive disclosure** — сначала высокоуровневая структура, потом детали
- **Teach through code** — комментарии объясняют математику и "зачем", а не только "что"

Все формы тензоров документированы: `# (T, D)`  
Вся нетривиальная математика объяснена: `# numerically-stable softmax`

## Литература

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — оригинальная статья про трансформеры
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — визуальное объяснение
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — autoregressive transformers
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — минималистичная реализация на PyTorch
