# Быстрый старт — PyTorch LLM

Минимальный гайд для запуска обучения за 5 минут.

## 0. Streamlit интерфейс

Самый простой способ — через веб-интерфейс:

```bash
streamlit run app.py
```

Откроется браузер с удобным UI для:
- **Генерации текста** — загрузите модель и генерируйте
- **Обучения в один клик** — настройте параметры → нажмите "Начать обучение"
  - Live прогресс: эпохи, train/val loss
  - Логи в реальном времени
  - Кнопка остановки
- **Загрузки данных** — drag-n-drop файлов
- **Информации** о модели и системе

**Не требует работы с терминалом!**

---

## 1. Установка

```bash
cd pytorch_llm
poetry add torch tqdm
```

Или с pip:
```bash
pip install torch tqdm
```

## 2. Создать данные

```bash
python train.py --prepare-sample
```

Создаст `data/sample.txt` с русским текстом (~2KB).

**Для реального обучения:** положите свой текст в `data/corpus.txt` (минимум 1MB).

## 3. Обучить модель

```bash
# Быстрое обучение (маленькая модель, 3 эпохи)
python train.py --config small --epochs 3 --batch-size 32

# Устройство определяется автоматически:
# - MPS на Apple Silicon (M1/M2/M3)
# - CUDA на NVIDIA GPU
# - CPU если нет GPU

# Явно указать устройство:
python train.py --config small --device mps   # macOS
python train.py --config small --device cuda  # NVIDIA
python train.py --config small --device cpu   # CPU
```

Время обучения:
- Apple Silicon (M1/M2 с MPS): ~5-10 минут
- NVIDIA GPU: ~5-10 минут
- CPU: ~30-60 минут

## 4. Генерация

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "Искусственный интеллект" \
  --max-tokens 100
```

## 5. Дообучение

```bash
# Добавьте свои данные и продолжите обучение
python train.py \
  --continue-from checkpoints/best_model.pt \
  --data data/my_new_data.txt \
  --epochs 5
```

---

## Полный пример в коде

```python
# example.py — запустите этот файл!
python example.py
```

Он сделает всё автоматически:
1. Создаст sample данные
2. Обучит модель (3 эпохи)
3. Сгенерирует текст с 3 промптами

---

## Параметры модели

Выберите размер в зависимости от ресурсов:

| Config | Параметры | GPU VRAM | Время обучения |
|--------|-----------|----------|----------------|
| **small** | ~13M | 4GB | 1 час (10 эпох) |
| **medium** | ~50M | 8GB | 4 часа (10 эпох) |
| **base** | ~117M | 16GB+ | 1-2 дня (10 эпох) |

---

## FAQ

**Q: "No module named 'torch'"**  
A: Установите PyTorch: `pip install torch` или `poetry add torch`

**Q: "CUDA not available" на macOS**  
A: Это нормально! На Mac используется MPS (Metal) вместо CUDA. Код автоматически определит правильное устройство.

**Q: "CUDA out of memory"**  
A: Уменьшите batch_size: `--batch-size 8` или используйте CPU: `--device cpu`

**Q: Модель генерирует бред**  
A: Нужно больше данных (минимум 1-10MB текста) и больше эпох (10-30)

**Q: Как использовать свои данные?**  
A: Положите текст в `data/my_data.txt` и запустите `python train.py --data data/my_data.txt`

---

## Следующие шаги

1. **Больше данных** — найдите корпус на 10-100MB (Wikipedia, книги, форумы)
2. **Больше эпох** — обучайте 10-30 эпох для лучшего качества
3. **Экспериментируйте** — попробуйте разные temperature (0.5-1.2) при генерации
4. **Дообучайте** — натренировали базовую модель? Дообучите на специфичных данных

Подробнее: [README.md](README.md)
