"""
Примеры использования Custom LLM API из разных языков.
"""

# ═══════════════════════════════════════════════════════════════
# Python (с OpenAI SDK)
# ═══════════════════════════════════════════════════════════════

from openai import OpenAI

# Инициализация клиента
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Не используется, но требуется
)

# Chat completion
def chat_example():
    response = client.chat.completions.create(
        model="custom-llm",
        messages=[
            {"role": "system", "content": "Ты — полезный ассистент."},
            {"role": "user", "content": "Расскажи о трансформерах в ML"}
        ],
        temperature=0.7,
        max_tokens=150
    )
    
    print(response.choices[0].message.content)

# Text completion
def completion_example():
    response = client.completions.create(
        model="custom-llm",
        prompt="Искусственный интеллект — это",
        temperature=0.8,
        max_tokens=100
    )
    
    print(response.choices[0].text)

# Streaming (если реализовано)
def streaming_example():
    stream = client.chat.completions.create(
        model="custom-llm",
        messages=[{"role": "user", "content": "Напиши стихотворение"}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

# ═══════════════════════════════════════════════════════════════
# Python (с requests)
# ═══════════════════════════════════════════════════════════════

import requests

def chat_with_requests():
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "custom-llm",
        "messages": [
            {"role": "user", "content": "Привет!"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    print(result["choices"][0]["message"]["content"])

# ═══════════════════════════════════════════════════════════════
# Python (LangChain интеграция)
# ═══════════════════════════════════════════════════════════════

from langchain.llms.openai import OpenAI as LangChainOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def langchain_example():
    # Используем ваш API как OpenAI
    llm = LangChainOpenAI(
        model_name="custom-llm",
        openai_api_base="http://localhost:8000/v1",
        openai_api_key="dummy"
    )
    
    # Создаём цепочку
    template = "Вопрос: {question}\n\nОтвет:"
    prompt = PromptTemplate(template=template, input_variables=["question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Выполняем
    result = chain.run(question="Что такое attention mechanism?")
    print(result)

# ═══════════════════════════════════════════════════════════════
# JavaScript / Node.js
# ═══════════════════════════════════════════════════════════════

"""
// package.json
{
  "dependencies": {
    "openai": "^4.0.0"
  }
}

// client.js
const OpenAI = require('openai');

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'dummy'
});

// Chat completion
async function chatExample() {
  const completion = await client.chat.completions.create({
    model: 'custom-llm',
    messages: [
      { role: 'user', content: 'Расскажи о нейронных сетях' }
    ],
    temperature: 0.7,
    max_tokens: 150
  });

  console.log(completion.choices[0].message.content);
}

// Text completion
async function completionExample() {
  const completion = await client.completions.create({
    model: 'custom-llm',
    prompt: 'Машинное обучение — это',
    temperature: 0.8,
    max_tokens: 100
  });

  console.log(completion.choices[0].text);
}

chatExample();
"""

# ═══════════════════════════════════════════════════════════════
# curl (bash)
# ═══════════════════════════════════════════════════════════════

"""
# Chat completion
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "custom-llm",
    "messages": [
      {"role": "user", "content": "Привет!"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# Text completion
curl http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "custom-llm",
    "prompt": "Искусственный интеллект — это",
    "temperature": 0.8,
    "max_tokens": 100
  }'

# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models
"""

# ═══════════════════════════════════════════════════════════════
# TypeScript (React app example)
# ═══════════════════════════════════════════════════════════════

"""
// api.ts
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'dummy',
  dangerouslyAllowBrowser: true  // Для браузера
});

export async function chat(message: string): Promise<string> {
  const completion = await client.chat.completions.create({
    model: 'custom-llm',
    messages: [{ role: 'user', content: message }],
    temperature: 0.7
  });

  return completion.choices[0].message.content || '';
}

// ChatComponent.tsx
import React, { useState } from 'react';
import { chat } from './api';

export function ChatComponent() {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    setLoading(true);
    const result = await chat(input);
    setResponse(result);
    setLoading(false);
  };

  return (
    <div>
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Введите сообщение"
      />
      <button onClick={handleSend} disabled={loading}>
        {loading ? 'Генерация...' : 'Отправить'}
      </button>
      <div>{response}</div>
    </div>
  );
}
"""

# ═══════════════════════════════════════════════════════════════
# Go
# ═══════════════════════════════════════════════════════════════

"""
// main.go
package main

import (
    "context"
    "fmt"
    "github.com/sashabaranov/go-openai"
)

func main() {
    config := openai.DefaultConfig("dummy")
    config.BaseURL = "http://localhost:8000/v1"
    client := openai.NewClientWithConfig(config)

    resp, err := client.CreateChatCompletion(
        context.Background(),
        openai.ChatCompletionRequest{
            Model: "custom-llm",
            Messages: []openai.ChatCompletionMessage{
                {
                    Role:    openai.ChatMessageRoleUser,
                    Content: "Привет!",
                },
            },
            Temperature: 0.7,
            MaxTokens:   100,
        },
    )

    if err != nil {
        fmt.Printf("Error: %v\\n", err)
        return
    }

    fmt.Println(resp.Choices[0].Message.Content)
}
"""

# ═══════════════════════════════════════════════════════════════
# PHP
# ═══════════════════════════════════════════════════════════════

"""
<?php
// composer require openai-php/client

require 'vendor/autoload.php';

use OpenAI\\Client;

$client = OpenAI::factory()
    ->withBaseUri('http://localhost:8000/v1')
    ->withApiKey('dummy')
    ->make();

$response = $client->chat()->create([
    'model' => 'custom-llm',
    'messages' => [
        ['role' => 'user', 'content' => 'Привет!'],
    ],
    'temperature' => 0.7,
    'max_tokens' => 100,
]);

echo $response->choices[0]->message->content;
"""

# ═══════════════════════════════════════════════════════════════
# Ruby
# ═══════════════════════════════════════════════════════════════

"""
# Gemfile: gem 'ruby-openai'

require 'openai'

client = OpenAI::Client.new(
  uri_base: 'http://localhost:8000/v1',
  access_token: 'dummy'
)

response = client.chat(
  parameters: {
    model: 'custom-llm',
    messages: [
      { role: 'user', content: 'Привет!' }
    ],
    temperature: 0.7,
    max_tokens: 100
  }
)

puts response.dig('choices', 0, 'message', 'content')
"""

# ═══════════════════════════════════════════════════════════════
# Java
# ═══════════════════════════════════════════════════════════════

"""
// Maven: com.theokanning.openai-gpt3-java

import com.theokanning.openai.service.OpenAiService;
import com.theokanning.openai.completion.chat.*;

public class CustomLLMClient {
    public static void main(String[] args) {
        OpenAiService service = new OpenAiService("dummy");
        service.setApiHost("http://localhost:8000/v1");

        ChatCompletionRequest request = ChatCompletionRequest.builder()
            .model("custom-llm")
            .messages(List.of(
                new ChatMessage("user", "Привет!")
            ))
            .temperature(0.7)
            .maxTokens(100)
            .build();

        ChatCompletionResult result = service.createChatCompletion(request);
        System.out.println(result.getChoices().get(0).getMessage().getContent());
    }
}
"""

# ═══════════════════════════════════════════════════════════════
# Примеры специализированных задач
# ═══════════════════════════════════════════════════════════════

def specialized_qa_bot():
    """Q&A бот для узкой области (например, медицина, право)."""
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )
    
    system_prompt = """Ты — эксперт в области машинного обучения. 
    Отвечай точно, лаконично, с примерами кода когда нужно."""
    
    def ask(question: str) -> str:
        response = client.chat.completions.create(
            model="custom-llm",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.3,  # Низкая temperature для точности
            max_tokens=300
        )
        return response.choices[0].message.content
    
    # Использование
    answer = ask("Что такое backpropagation?")
    print(answer)

def chat_with_history():
    """Чат с сохранением контекста."""
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"
    )
    
    history = [
        {"role": "system", "content": "Ты — дружелюбный ассистент."}
    ]
    
    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        history.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="custom-llm",
            messages=history,
            temperature=0.7
        )
        
        assistant_response = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_response})
        
        print(f"Ассистент: {assistant_response}")

if __name__ == "__main__":
    print("Примеры использования Custom LLM API")
    print("=" * 50)
    
    # Запустите нужный пример:
    # chat_example()
    # completion_example()
    # langchain_example()
    # specialized_qa_bot()
    # chat_with_history()
