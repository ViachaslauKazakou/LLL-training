# 🌐 Deployment и интеграция вашей модели

## 🎯 Варианты использования

### 1. ✅ OpenAI-compatible API (РЕКОМЕНДУЕТСЯ)

**Преимущества:**
- Совместимость с OpenAI SDK и библиотеками
- Легко интегрировать в существующие приложения
- Стандартный REST API
- Работает с LangChain, LlamaIndex и другими

**Запуск сервера:**
```bash
cd pytorch_llm
poetry add fastapi uvicorn pydantic
poetry run uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**Использование (curl):**
```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom-llm",
    "messages": [
      {"role": "user", "content": "Привет! Как дела?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# Text completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom-llm",
    "prompt": "Искусственный интеллект — это",
    "temperature": 0.8,
    "max_tokens": 100
  }'
```

**Использование (Python с OpenAI SDK):**
```python
from openai import OpenAI

# Подключаемся к вашему серверу
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # не используется, но требуется SDK
)

# Используем как OpenAI!
response = client.chat.completions.create(
    model="custom-llm",
    messages=[
        {"role": "user", "content": "Расскажи о машинном обучении"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

**Использование (JavaScript):**
```javascript
const OpenAI = require('openai');

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'dummy'
});

async function main() {
  const completion = await client.chat.completions.create({
    model: 'custom-llm',
    messages: [
      { role: 'user', content: 'Привет!' }
    ],
    temperature: 0.7
  });

  console.log(completion.choices[0].message.content);
}

main();
```

**Деплой на production:**
```bash
# С Docker
docker build -t custom-llm-api .
docker run -p 8000:8000 custom-llm-api

# С systemd
sudo systemctl start custom-llm-api

# С PM2 (Node.js)
pm2 start "uvicorn api_server:app --host 0.0.0.0 --port 8000"
```

---

### 2. ⚠️ Ollama (требует конвертации)

**Проблема:** Ollama использует формат GGUF (quantized), а ваша модель в PyTorch .pt

**Вариант A: Конвертация через GGUF (сложно)**

Ollama работает с моделями в формате GGUF (llama.cpp). Для конвертации нужно:

1. **Конвертировать PyTorch → GGUF:**
   ```bash
   # Требуется llama.cpp
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   
   # Конвертация (но требуется совместимая архитектура!)
   python convert.py /path/to/your/model
   
   # Квантизация
   ./quantize model.gguf model-q4_0.gguf q4_0
   ```

2. **Создать Modelfile для Ollama:**
   ```dockerfile
   # Modelfile
   FROM ./model-q4_0.gguf
   
   PARAMETER temperature 0.7
   PARAMETER top_k 50
   
   TEMPLATE """{{ .Prompt }}"""
   
   SYSTEM """You are a helpful AI assistant."""
   ```

3. **Импорт в Ollama:**
   ```bash
   ollama create custom-llm -f Modelfile
   ollama run custom-llm "Привет!"
   ```

**❌ Минусы:**
- Сложная конвертация
- Ваша архитектура может быть несовместима
- Потеря качества при квантизации
- Требуется переобучение для оптимизации

**Вариант B: Ollama как прокси (проще)**

Используйте Ollama только как фронтенд, а модель держите на вашем API:

```bash
# Запустите ваш API сервер
uvicorn api_server:app --port 8000

# Настройте Ollama для проксирования
# (требует custom Ollama plugin)
```

---

### 3. 🚀 LangChain интеграция

```python
from langchain.llms.base import LLM
from typing import Optional, List
import requests

class CustomLLM(LLM):
    endpoint: str = "http://localhost:8000/v1/completions"
    temperature: float = 0.7
    max_tokens: int = 100
    
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None
    ) -> str:
        response = requests.post(
            self.endpoint,
            json={
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        )
        return response.json()["choices"][0]["text"]

# Использование
llm = CustomLLM()
result = llm("Расскажи о нейронных сетях")
```

---

### 4. 📱 Streamlit Share (для демо)

Разверните ваш Streamlit UI публично:

```bash
# 1. Создайте requirements.txt
cat > requirements.txt << EOF
streamlit>=1.45.0
torch>=2.0.0
tqdm
EOF

# 2. Deploy на Streamlit Cloud
# - Залейте на GitHub
# - Подключите через streamlit.io/cloud
# - Готово! Публичный URL для демо
```

---

## 🏢 Production deployment

### Docker контейнер

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Зависимости
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

# Код и модель
COPY pytorch_llm/ ./pytorch_llm/
COPY checkpoints/best_model.pt ./checkpoints/

# Порт
EXPOSE 8000

# Запуск
CMD ["poetry", "run", "uvicorn", "pytorch_llm.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Сборка
docker build -t custom-llm-api .

# Запуск
docker run -p 8000:8000 -e DEVICE=cuda custom-llm-api

# С GPU
docker run --gpus all -p 8000:8000 custom-llm-api
```

### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: custom-llm-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: custom-llm
  template:
    metadata:
      labels:
        app: custom-llm
    spec:
      containers:
      - name: api
        image: custom-llm-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: custom-llm-service
spec:
  selector:
    app: custom-llm
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

```bash
kubectl apply -f deployment.yaml
```

### systemd service (Linux)

```ini
# /etc/systemd/system/custom-llm.service
[Unit]
Description=Custom LLM API Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/custom-llm
ExecStart=/opt/custom-llm/.venv/bin/uvicorn pytorch_llm.api_server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable custom-llm
sudo systemctl start custom-llm
sudo systemctl status custom-llm
```

---

## 🔒 Безопасность

### 1. API ключи

Добавьте аутентификацию:

```python
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()
API_KEY = "your-secret-key-here"

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials

# В эндпоинтах
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    token: str = Depends(verify_token)
):
    # ...
```

### 2. Rate limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/v1/chat/completions")
@limiter.limit("10/minute")
async def chat_completions(request: Request, ...):
    # ...
```

### 3. HTTPS

```bash
# С Nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
    }
}

# Или с Certbot (Let's Encrypt)
sudo certbot --nginx -d api.yourdomain.com
```

---

## 📊 Мониторинг

### Prometheus metrics

```python
from prometheus_client import Counter, Histogram, make_asgi_app

# Метрики
requests_total = Counter('requests_total', 'Total requests')
latency = Histogram('latency_seconds', 'Request latency')

# Эндпоинт метрик
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### Логирование

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    logger.info(f"Request: {request.messages[-1].content[:50]}")
    # ...
    logger.info(f"Response generated: {len(response_text)} chars")
```

---

## ⚡ Оптимизация производительности

### 1. Батчинг запросов

```python
from asyncio import Queue, create_task

batch_queue = Queue()

async def batch_processor():
    while True:
        batch = []
        for _ in range(8):  # batch size
            if not batch_queue.empty():
                batch.append(await batch_queue.get())
        
        if batch:
            # Обработка батча
            results = model.generate_batch(batch)
            # Возврат результатов
```

### 2. Кэширование

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def generate_cached(prompt: str, temp: float, max_tokens: int):
    return generate_text(MODEL, TOKENIZER, prompt, ...)
```

### 3. Квантизация модели

```python
import torch.quantization

# INT8 quantization
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Размер: 37.5 MB → 10 MB
# Скорость: 2-3x быстрее
# Качество: -1-2% loss
```

---

## 🎯 Рекомендации

### Для начала:
✅ **OpenAI-compatible API** через FastAPI  
✅ Docker контейнер  
✅ systemd service на сервере

### Если нужна масштабируемость:
✅ Kubernetes  
✅ Load balancer  
✅ Horizontal scaling

### Если нужна интеграция:
✅ LangChain custom LLM  
✅ REST API для любых языков  
✅ WebSocket для streaming

### ❌ НЕ рекомендую:
- Ollama (сложная конвертация, может не сработать)
- Прямое использование .pt файла на клиенте (требует PyTorch)

---

## 🚀 Быстрый старт

```bash
# 1. Установка зависимостей
cd pytorch_llm
poetry add fastapi uvicorn pydantic

# 2. Запуск API сервера
poetry run uvicorn api_server:app --host 0.0.0.0 --port 8000

# 3. Тест
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"custom-llm","messages":[{"role":"user","content":"Привет!"}]}'

# 4. Production deploy (Docker)
docker build -t custom-llm-api .
docker run -p 8000:8000 custom-llm-api
```

**Готово!** Ваша модель теперь доступна через OpenAI-compatible API! 🎉
