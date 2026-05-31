"""
API Server для модели с OpenAI-compatible интерфейсом.

Запуск:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Использование:
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "custom-llm",
        "messages": [{"role": "user", "content": "Привет!"}],
        "temperature": 0.7
      }'
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from pathlib import Path
import time

from inference import load_model_for_inference, generate_text
from data import CharTokenizer
from config import get_device

# ═══════════════════════════════════════════════════════════════
# Модели данных (OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "custom-llm"
    messages: List[Message]
    temperature: Optional[float] = 0.8
    max_tokens: Optional[int] = 100
    top_k: Optional[int] = 50
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

class CompletionRequest(BaseModel):
    model: str = "custom-llm"
    prompt: str
    temperature: Optional[float] = 0.8
    max_tokens: Optional[int] = 100
    top_k: Optional[int] = 50

# ═══════════════════════════════════════════════════════════════
# Инициализация
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Custom LLM API",
    description="OpenAI-compatible API для вашей обученной модели",
    version="1.0.0"
)

# CORS для доступа из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для модели
MODEL = None
TOKENIZER = None
DEVICE = None
CHECKPOINT_PATH = "checkpoints/best_model.pt"
DATA_PATH = "data/sample.txt"  # Для токенизатора

# ═══════════════════════════════════════════════════════════════
# Загрузка модели при старте
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def load_model():
    """Загружает модель при запуске сервера."""
    global MODEL, TOKENIZER, DEVICE
    
    print(f"🔄 Загрузка модели из {CHECKPOINT_PATH}...")
    
    DEVICE = get_device()
    MODEL, checkpoint = load_model_for_inference(CHECKPOINT_PATH, DEVICE)
    
    # Создаём токенизатор
    text = Path(DATA_PATH).read_text(encoding='utf-8')
    TOKENIZER = CharTokenizer(text)
    
    print(f"✅ Модель загружена на {DEVICE}")
    print(f"   Параметров: {MODEL.count_parameters():,}")
    print(f"   Vocab size: {len(TOKENIZER.vocab)}")

# ═══════════════════════════════════════════════════════════════
# Эндпоинты
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Информация об API."""
    return {
        "name": "Custom LLM API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": MODEL is not None,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completion": "/v1/completions",
            "models": "/v1/models"
        }
    }

@app.get("/v1/models")
async def list_models():
    """Список доступных моделей (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "custom-llm",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
                "permission": [],
                "root": "custom-llm",
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    
    Пример запроса:
        {
          "model": "custom-llm",
          "messages": [
            {"role": "user", "content": "Расскажи о машинном обучении"}
          ],
          "temperature": 0.7,
          "max_tokens": 150
        }
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Формируем промпт из истории сообщений
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    
    # Добавляем префикс для ответа ассистента
    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)
    
    # Генерация
    try:
        generated = generate_text(
            MODEL,
            TOKENIZER,
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            device=DEVICE
        )
        
        # Извлекаем только ответ ассистента
        response_text = generated.split("Assistant:")[-1].strip()
        
        # Формируем ответ в формате OpenAI
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": len(prompt),
                "completion_tokens": len(response_text),
                "total_tokens": len(prompt) + len(response_text)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    OpenAI-compatible completions endpoint.
    
    Пример запроса:
        {
          "model": "custom-llm",
          "prompt": "Искусственный интеллект — это",
          "temperature": 0.8,
          "max_tokens": 100
        }
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        generated = generate_text(
            MODEL,
            TOKENIZER,
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            device=DEVICE
        )
        
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "text": generated,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(request.prompt),
                "completion_tokens": len(generated) - len(request.prompt),
                "total_tokens": len(generated)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }

# ═══════════════════════════════════════════════════════════════
# Запуск
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
