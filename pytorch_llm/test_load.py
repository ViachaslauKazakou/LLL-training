#!/usr/bin/env python3
"""Test checkpoint loading with PyTorch 2.6 compatibility"""
import torch
import sys
from pathlib import Path

checkpoint_path = Path("checkpoints/best_model.pt")

if not checkpoint_path.exists():
    print(f"❌ Checkpoint не найден: {checkpoint_path}")
    sys.exit(1)

print(f"PyTorch версия: {torch.__version__}")
print(f"Загрузка checkpoint: {checkpoint_path}")

try:
    # Загружаем с weights_only=False (наши файлы безопасны)
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print("\n✅ Checkpoint загружен успешно!")
    print(f"\nМетрики:")
    print(f"  Global step: {cp.get('global_step', 'N/A')}")
    
    val_loss = cp.get('best_val_loss', float('inf'))
    if val_loss != float('inf'):
        print(f"  Best val loss: {val_loss:.4f}")
    
    config = cp['config']
    print(f"\nКонфигурация модели:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    
    print("\n✅ Все проверки пройдены!")
    print("Модель готова к использованию в Streamlit UI")
    
except Exception as e:
    print(f"\n❌ Ошибка загрузки: {e}")
    sys.exit(1)
