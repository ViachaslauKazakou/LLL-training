# PyTorch LLM — полноценный трансформер с нуля

Авторегрессионный трансформер (GPT-подобная архитектура) на PyTorch с возможностью обучения и дообучения.

## Особенности

✅ **Полная архитектура трансформера**
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Residual connections
- Positional embeddings

✅ **Production-ready training**
- AdamW optimizer с weight decay
- Cosine learning rate schedule
- Gradient clipping
- Checkpointing
- Validation мониторинг

✅ **Гибкая конфигурация**
- 3 предустановленных размера (small/medium/base)
- Все гиперпараметры настраиваемые

✅ **Дообучение (fine-tuning)**
- Загрузка checkpoint
- Продолжение обучения с сохранённого состояния

---

## Установка

```bash
# Зависимости
pip install torch tqdm

# Или через poetry
poetry add torch tqdm
```

---

## Быстрый старт

### 0. Streamlit интерфейс (рекомендуется)

```bash
# Запустить веб-интерфейс
streamlit run app.py
```

Streamlit UI включает:
- 🎯 **Генерация** — интерактивная генерация текста с настройкой параметров
- 🎓 **Обучение** — запуск обучения прямо из браузера с live прогрессом
- 📁 **Данные** — загрузка и подготовка данных
- ℹ️ **Инфо** — документация и системная информация

**Обучение через UI:**
- Настройка всех параметров в браузере
- Кнопка "Начать обучение" — запуск одним кликом
- Live отображение прогресса, метрик и логов
- Возможность остановить обучение
- Не требует работы с терминалом

### 1. Подготовка данных

```bash
# Создать sample датасет для тестирования
python train.py --prepare-sample

# Или положите свой текст в data/corpus.txt
```

### 2. Обучение с нуля

```bash
# Маленькая модель (~13M параметров)
python train.py --config small --epochs 10 --batch-size 32

# Средняя модель (~50M параметров)
python train.py --config medium --epochs 5 --batch-size 16

# Базовая модель (~117M параметров, как GPT-2 small)
python train.py --config base --epochs 3 --batch-size 8

# На конкретном устройстве
python train.py --config small --device mps   # Apple Silicon
python train.py --config small --device cuda  # NVIDIA GPU
python train.py --config small --device cpu   # CPU
```

**Параметры:**
- `--data` — путь к текстовому файлу
- `--config` — размер модели (small/medium/base)
- `--epochs` — количество эпох
- `--batch-size` — размер батча
- `--lr` — learning rate (default: 3e-4)
- `--device` — auto/mps/cuda/cpu (default: auto — автоопределение)

### 3. Генерация текста

```bash
# Разовая генерация
python inference.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "Искусственный интеллект" \
  --max-tokens 200 \
  --temperature 0.8

# Интерактивный режим
python inference.py \
  --checkpoint checkpoints/best_model.pt \
  --interactive
```

**Параметры генерации:**
- `temperature` — креативность (0.1 = детерминированно, 2.0 = очень случайно)
- `top_k` — выбирать из top-k токенов (50 = хороший баланс)

### 4. Дообучение (fine-tuning)

```bash
# Продолжить обучение с checkpoint
python train.py \
  --continue-from checkpoints/best_model.pt \
  --epochs 5 \
  --data data/new_corpus.txt
```

---

## Архитектура

### Структура модели

```
Input Tokens (batch, seq_len)
     ↓
Token Embedding (vocab_size → d_model)
     +
Positional Embedding (context_len → d_model)
     ↓
Dropout
     ↓
N × Transformer Block:
│   LayerNorm
│   ↓
│   Multi-Head Self-Attention (n_heads)
│   ↓
│   Residual Connection
│   ↓
│   LayerNorm
│   ↓
│   Feed-Forward Network (d_model → 4×d_model → d_model)
│   ↓
│   Residual Connection
     ↓
LayerNorm
     ↓
Linear (d_model → vocab_size)
     ↓
Softmax → Probabilities
```

### Размеры моделей

| Конфиг | Параметры | d_model | n_layers | n_heads | context_len |
|--------|-----------|---------|----------|---------|-------------|
| **small** | ~13M | 256 | 4 | 4 | 256 |
| **medium** | ~50M | 512 | 6 | 8 | 512 |
| **base** | ~117M | 768 | 12 | 12 | 1024 |

├── app.py             # Streamlit веб-интерфейс
├── example.py         # Полный пример использования
---

## Структура проекта

```
pytorch_llm/
├── config.py          # Конфигурации (ModelConfig, TrainingConfig)
├── attention.py       # Multi-head self-attention + causal mask
├── model.py           # GPTModel — полная архитектура
├── data.py            # Dataset, DataLoader, токенизатор
├── training.py        # Trainer — цикл обучения
├── train.py           # Entry point для обучения
├── inference.py       # Генерация текста
└── README.md

checkpoints/           # Сохранённые модели
data/                  # Данные для обучения
```

---

## Примеры использования

### Обучить на своих данных

```python
from config import get_small_config, TrainingConfig
from data import load_data
from model import GPTModel
from training import Trainer

# Конфиг
model_config = get_small_config()
train_config = TrainingConfig(
    n_epochs=10,
    data_path="my_corpus.txt"
)

# Загрузка данных
train_loader, val_loader, tokenizer = load_data(
    "my_corpus.txt",
    model_config.context_len,
    model_config.batch_size
)

# Обновляем vocab_size из данных
model_config.vocab_size = len(tokenizer.vocab)

# Создаём модель
model = GPTModel(model_config)

# Обучаем
trainer = Trainer(model, train_loader, val_loader, train_config)
trainer.train()
```

### Дообучение существующей модели

```python
from training import continue_training

continue_training(
    checkpoint_path="checkpoints/best_model.pt",
    train_loader=new_train_loader,
    val_loader=new_val_loader,
    config=train_config,
    additional_epochs=5
)
```

### Генерация программно

```python
import torch
from inference import load_model_for_inference, generate_text
from data import CharTokenizer

# Загрузка
model, checkpoint = load_model_for_inference("checkpoints/best_model.pt")
tokenizer = CharTokenizer(open("data/sample.txt").read())

# Генерация
text = generate_text(
    model, 
    tokenizer,
    prompt="Машинное обучение",
    max_new_tokens=100,
    temperature=0.7,
    top_k=50
)

print(text)
```

---

## Требования

### Поддержка GPU

**🍎 macOS (Apple Silicon)**
- ✅ MPS (Metal Performance Shaders) — встроенный GPU
- Обучение в 5-10 раз быстрее чем на CPU
- Автоматически определяется при `--device auto`

**🎮 Windows/Linux (NVIDIA)**
- ✅ CUDA — NVIDIA GPU
- Требуется CUDA Toolkit + CUDA-enabled PyTorch
- Установка: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**💻 Fallback на CPU**
- Работает везде, но медленно (в 10-50 раз)
- Подходит для небольших экспериментов

### Минимальные (small config)
- GPU: 4GB VRAM (GTX 1050 Ti)
- RAM: 8GB
- Время обучения: ~1 час на 10 эпох

### Рекомендуемые (medium config)
- GPU: 8GB VRAM (RTX 3060)
- RAM: 16GB
- Время обучения: ~4 часа на 10 эпох

### Для base config
- GPU: 16GB+ VRAM (RTX 3090 / A100 / M2 Max 32GB)
- RAM: 32GB+
- Время обучения: ~1-2 дня на 10 эпох

**На CPU:** возможно, но медленно (в 10-50 раз)

---

## Дальнейшие улучшения

Текущая реализация — solid baseline. Что можно добавить:

### Performance
- [ ] Flash Attention (в 2-4 раза быстрее)
- [ ] Gradient checkpointing (меньше памяти)
- [ ] Mixed precision training (FP16/BF16)
- [ ] Distributed training (multi-GPU)

### Tokenization
- [ ] BPE токенизатор (tiktoken)
- [ ] SentencePiece
- [ ] Hugging Face tokenizers

### Архитектура
- [ ] Rotary Positional Embeddings (RoPE)
- [ ] Flash Attention v2
- [ ] Group Query Attention (GQA)

### Training
- [ ] Wandb/TensorBoard интеграция
- [ ] Gradient accumulation (больший эффективный batch)
- [ ] Learning rate warmup
- [ ] Label smoothing

### Inference
- [ ] Beam search
- [ ] Nucleus (top-p) sampling
- [ ] Repetition penalty
- [ ] KV-cache для ускорения

---

## Сравнение с Mini LLM

| Фича | Mini LLM (NumPy) | PyTorch LLM |
|------|------------------|-------------|
| **Attention** | Single-head | Multi-head (8-12 голов) |
| **Layers** | 1 | 4-12 слоёв |
| **FFN** | ❌ Нет | ✅ Есть |
| **LayerNorm** | ❌ Нет | ✅ Есть |
| **Residual** | ❌ Нет | ✅ Есть |
| **Параметры** | ~6K | 13M-117M |
| **Context** | 4-8 токенов | 256-1024 токена |
| **Обучение** | Manual backward | PyTorch autograd |
| **Скорость** | Медленно | Быстро (GPU) |
| **Результат** | Игрушка | Работающая LLM |

---

## FAQ

**Q: Сколько времени займёт обучение?**  
A: Small config — 1 час, medium — 4 часа, base — 1-2 дня (на RTX 3090 или M2 Max).

**Q: Можно ли обучить без GPU?**  
A: Да, но будет в 10-50 раз медленнее. Для экспериментов хватит.

**Q: "CUDA не обнаружен" на macOS**  
A: Это нормально! На Mac используется MPS (Apple GPU) вместо CUDA. Код автоматически выберет правильное устройство.

**Q: Как включить MPS на Mac?**  
A: MPS включается автоматически при `--device auto`. Или явно: `--device mps`

**Q: Какой размер данных нужен?**  
A: Минимум 1MB текста (~500K токенов). Хорошо — 10-100MB. Идеально — 1GB+.

**Q: Можно ли дообучить на другом языке?**  
A: Да, просто продолжите обучение с новым датасетом.

**Q: Как улучшить качество?**  
A: 1) Больше данных, 2) Больше эпох, 3) Больше параметров модели.

---

## Лицензия

MIT
