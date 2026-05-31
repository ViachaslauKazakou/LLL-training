# 🚀 Quick Start: Запуск API за 5 минут

## Шаг 1: Установка зависимостей

```bash
cd /Users/Viachaslau_Kazakou/Work/LLM-learn/pytorch_llm
poetry add fastapi uvicorn pydantic
```

## Шаг 2: Запуск сервера

```bash
poetry run uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Вы увидите:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
🔄 Загрузка модели из checkpoints/best_model.pt...
✅ Модель загружена на mps
   Параметров: 3,268,864
   Vocab size: 171
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Шаг 3: Тестирование

### Вариант A: Через curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom-llm",
    "messages": [
      {"role": "user", "content": "Привет! Расскажи о нейронных сетях"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

### Вариант B: Через Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="custom-llm",
    messages=[
        {"role": "user", "content": "Привет!"}
    ]
)

print(response.choices[0].message.content)
```

### Вариант C: Через браузер

Откройте http://localhost:8000/docs — интерактивная документация (Swagger UI)

## Шаг 4: Интеграция в приложение

Теперь ваш API совместим с любым приложением, которое использует OpenAI!

```python
# Просто измените base_url
from openai import OpenAI

# Было:
# client = OpenAI(api_key="sk-...")

# Стало:
client = OpenAI(
    base_url="http://your-server:8000/v1",
    api_key="dummy"
)

# Весь остальной код работает без изменений!
```

## Готово! 🎉

Ваша модель теперь доступна через стандартный OpenAI API и может использоваться:
- ✅ В веб-приложениях (JavaScript, React, Vue, etc)
- ✅ В мобильных приложениях (iOS, Android)
- ✅ В десктопных приложениях (Electron, Qt, etc)
- ✅ С LangChain, LlamaIndex и другими фреймворками
- ✅ С любыми библиотеками, поддерживающими OpenAI API

---

## Следующие шаги:

1. **Production deployment:** См. [DEPLOYMENT.md](DEPLOYMENT.md)
2. **Примеры клиентов:** См. [client_examples.py](../examples/client_examples.py)
3. **Безопасность:** Добавьте API ключи и rate limiting
4. **Мониторинг:** Настройте логирование и метрики
