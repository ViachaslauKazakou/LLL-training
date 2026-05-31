#!/usr/bin/env python3
"""Тест подготовки корпуса"""
import pathlib
import re
import json
from collections import Counter

input_path = pathlib.Path('example_dialog.txt')
output_path = pathlib.Path('corpus_dialog.json')

with input_path.open(encoding='utf-8') as f:
    text = f.read()

# Разбиваем на предложения
sentences = re.split(r'[.!?\n]+', text)
sentences = [re.sub(r'\s+', ' ', s).strip() for s in sentences if s.strip()]

# Фильтруем по длине (2-15 слов)
filtered = []
for s in sentences:
    words = s.split()
    if 2 <= len(words) <= 15 and len(s) >= 5:
        filtered.append(s)

# Убираем дубликаты с сохранением частоты
counts = Counter(filtered)
expanded = []
for sent, count in counts.items():
    expanded.extend([sent] * min(count, 3))

corpus = {'sentences': expanded}

with output_path.open('w', encoding='utf-8') as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

print(f'✓ Обработано: {len(filtered)} уникальных предложений')
print(f'✓ В корпусе: {len(expanded)} предложений')
print(f'✓ Словарь: {len(set(" ".join(expanded).split()))} слов')
print(f'✓ Сохранено: {output_path}')

print(f'\nПервые 5 предложений:')
for i, s in enumerate(expanded[:5], 1):
    print(f'  {i}. {s}')

# Обучаем и тестируем
print('\n=== Обучение на диалоге ===')
from model import MiniLLM
llm = MiniLLM(d_model=24, context_len=5)
llm.load_corpus('corpus_dialog.json')
llm.train(n_epochs=200, lr=0.01)
llm.save('saved/dialog_demo.npz')

print('\n=== Генерация ===')
prompts = ['Привет', 'Как дела', 'Окей']
for p in prompts:
    result = llm.generate(p, n_new=5, temperature=0.6, repetition_penalty=1.3, seed=0)
    print(f'  {p:15} → {result}')
