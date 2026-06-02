#!/usr/bin/env python3
"""
Показывает что находится внутри checkpoint файла
"""
import torch
from pathlib import Path
from logger import app_logger

checkpoint_path = "checkpoints/best_model.pt"

app_logger.info("="*70)
app_logger.info("🔍 ЧТО ХРАНИТСЯ В CHECKPOINT ФАЙЛЕ")
app_logger.info("="*70)

# Загружаем checkpoint
cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

app_logger.info(f"\n📦 Файл: {checkpoint_path}")
app_logger.info(f"💾 Размер: {Path(checkpoint_path).stat().st_size / 1024 / 1024:.2f} MB")

app_logger.info("\n📋 Содержимое checkpoint:")
app_logger.info("-" * 70)

for key in cp.keys():
    value = cp[key]
    if isinstance(value, dict):
        app_logger.info(f"  {key}: dict с {len(value)} элементами")
        if key == 'model_state_dict':
            total_params = sum(p.numel() for p in value.values())
            app_logger.info(f"    └─ Всего параметров: {total_params:,}")
    elif hasattr(value, '__class__'):
        app_logger.info(f"  {key}: {value.__class__.__name__}")
    else:
        app_logger.info(f"  {key}: {value}")

app_logger.info("\n" + "="*70)
app_logger.info("🧠 ВЕСА МОДЕЛИ (model_state_dict)")
app_logger.info("="*70)

state_dict = cp['model_state_dict']
app_logger.info(f"\nВсего слоёв: {len(state_dict)}")
app_logger.info("\nПервые 10 слоёв:")
for i, (name, tensor) in enumerate(list(state_dict.items())[:10]):
    app_logger.info(f"  {name:50s} shape: {str(tuple(tensor.shape)):20s} ({tensor.numel():,} параметров)")

app_logger.info("\n" + "="*70)
app_logger.info("⚙️ КОНФИГУРАЦИЯ МОДЕЛИ (config)")
app_logger.info("="*70)

config = cp['config']
app_logger.info(f"\nАрхитектура:")
app_logger.info(f"  vocab_size:  {config.vocab_size:,}")
app_logger.info(f"  d_model:     {config.d_model}")
app_logger.info(f"  n_layers:    {config.n_layers}")
app_logger.info(f"  n_heads:     {config.n_heads}")
app_logger.info(f"  d_ff:        {config.d_ff}")
app_logger.info(f"  context_len: {config.context_len}")
app_logger.info(f"  dropout:     {config.dropout}")

app_logger.info("\n" + "="*70)
app_logger.info("📊 СОСТОЯНИЕ ОБУЧЕНИЯ")
app_logger.info("="*70)

app_logger.info(f"\nГлобальный шаг: {cp.get('global_step', 'N/A')}")
val_loss = cp.get('best_val_loss', float('inf'))
if val_loss != float('inf'):
    app_logger.info(f"Лучший val_loss: {val_loss:.4f}")
    import math
    app_logger.info(f"Perplexity: {math.exp(val_loss):.2f}")

app_logger.info("\n" + "="*70)
app_logger.info("💡 КАК ИСПОЛЬЗОВАТЬ")
app_logger.info("="*70)
app_logger.info("""
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

app_logger.info("="*70)
