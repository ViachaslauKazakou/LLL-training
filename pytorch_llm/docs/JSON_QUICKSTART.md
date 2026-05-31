# Быстрый старт с JSON данными

## 1. Подготовить JSON файл

Формат:
```json
{
  "User": "123456",
  "messages": {
    "Тема 1": ["сообщение 1", "сообщение 2"],
    "Тема 2": ["сообщение 1"]
  }
}
```

## 2. Обучить модель

```bash
cd pytorch_llm

# Обучение
poetry run python train.py \
  --data data/your_forum.json \
  --config small \
  --epochs 30 \
  --batch-size 16

# Или через UI
poetry run streamlit run app.py
```

## 3. Генерация

```bash
poetry run python inference.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "Ваш промпт" \
  --interactive
```

Подробнее: [JSON_SUPPORT.md](./JSON_SUPPORT.md)
