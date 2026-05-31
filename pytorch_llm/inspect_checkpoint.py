#!/usr/bin/env python3
"""
Показывает что находится внутри checkpoint файла
"""
import torch
from pathlib import Path

checkpoint_path = "checkpoints/best_model.pt"

print("="*70)
print("🔍 ЧТО ХРАНИТСЯ В CHECKPOINT ФАЙЛЕ")
print("="*70)

# Загружаем checkpoint
cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print(f"\n📦 Файл: {checkpoint_path}")
print(f"💾 Размер: {Path(checkpoint_path).stat().st_size / 1024 / 1024:.2f} MB")

print("\n📋 Содержимое checkpoint:")
print("-" * 70)

for key in cp.keys():
    value = cp[key]
    if isinstance(value, dict):
        print(f"  {key}: dict с {len(value)} элементами")
        if key == 'model_state_dict':
            total_params = sum(p.numel() for p in value.values())
            print(f"    └─ Всего параметров: {total_params:,}")
    elif hasattr(value, '__class__'):
        print(f"  {key}: {value.__class__.__name__}")
    else:
        print(f"  {key}: {value}")

print("\n" + "="*70)
print("🧠 ВЕСА МОДЕЛИ (model_state_dict)")
print("="*70)

state_dict = cp['model_state_dict']
print(f"\nВсего слоёв: {len(state_dict)}")
print("\nПервые 10 слоёв:")
for i, (name, tensor) in enumerate(list(state_dict.items())[:10]):
    print(f"  {name:50s} shape: {str(tuple(tensor.shape)):20s} ({tensor.numel():,} параметров)")

print("\n" + "="*70)
print("⚙️ КОНФИГУРАЦИЯ МОДЕЛИ (config)")
print("="*70)

config = cp['config']
print(f"\nАрхитектура:")
print(f"  vocab_size:  {config.vocab_size:,}")
print(f"  d_model:     {config.d_model}")
print(f"  n_layers:    {config.n_layers}")
print(f"  n_heads:     {config.n_heads}")
print(f"  d_ff:        {config.d_ff}")
print(f"  context_len: {config.context_len}")
print(f"  dropout:     {config.dropout}")

print("\n" + "="*70)
print("📊 СОСТОЯНИЕ ОБУЧЕНИЯ")
print("="*70)

print(f"\nГлобальный шаг: {cp.get('global_step', 'N/A')}")
val_loss = cp.get('best_val_loss', float('inf'))
if val_loss != float('inf'):
    print(f"Лучший val_loss: {val_loss:.4f}")
    import math
    print(f"Perplexity: {math.exp(val_loss):.2f}")

print("\n" + "="*70)
print("💡 КАК ИСПОЛЬЗОВАТЬ")
print("="*70)
print("""
Этот checkpoint содержит ВСЁ необходимое для:

1. ✅ Генерации текста (inference):
   - Архитектура модели (config)
   - Обученные веса (model_state_dict)

2. ✅ Продолжения обучения (fine-tuning):
   - Состояние optimizer (optimizer_state_dict)
   - Состояние scheduler (scheduler_state_dict)
   - Текущий шаг обучения (global_step)

3. ✅ Переноса на другой компьютер:
   - Просто скопируйте .pt файл
   - Загрузите с теми же ModelConfig
""")

print("="*70)
