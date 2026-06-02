#!/usr/bin/env python3
"""Test checkpoint loading with PyTorch 2.6 compatibility"""
import torch
import sys
from pathlib import Path
from logger import app_logger

checkpoint_path = Path("checkpoints/best_model.pt")

if not checkpoint_path.exists():
    app_logger.info(f"❌ Checkpoint не найден: {checkpoint_path}")
    sys.exit(1)

app_logger.info(f"PyTorch версия: {torch.__version__}")
app_logger.info(f"Загрузка checkpoint: {checkpoint_path}")

try:
    # Загружаем с weights_only=False (наши файлы безопасны)
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    app_logger.info("\n✅ Checkpoint загружен успешно!")
    app_logger.info(f"\nМетрики:")
    app_logger.info(f"  Global step: {cp.get('global_step', 'N/A')}")
    
    val_loss = cp.get('best_val_loss', float('inf'))
    if val_loss != float('inf'):
        app_logger.info(f"  Best val loss: {val_loss:.4f}")
    
    config = cp['config']
    app_logger.info(f"\nКонфигурация модели:")
    app_logger.info(f"  vocab_size: {config.vocab_size}")
    app_logger.info(f"  d_model: {config.d_model}")
    app_logger.info(f"  n_layers: {config.n_layers}")
    
    app_logger.info("\n✅ Все проверки пройдены!")
    app_logger.info("Модель готова к использованию в Streamlit UI")
    
except Exception as e:
    app_logger.info(f"\n❌ Ошибка загрузки: {e}")
    sys.exit(1)
