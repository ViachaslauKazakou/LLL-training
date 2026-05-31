"""
Скрипт для сравнения checkpoint'ов и выбора лучшего.
"""

import torch
from pathlib import Path
import sys


def inspect_checkpoint(checkpoint_path: str):
    """Показывает информацию о checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print(f"\n{'='*60}")
        print(f"Checkpoint: {Path(checkpoint_path).name}")
        print(f"{'='*60}")
        
        # Основная информация
        if 'global_step' in checkpoint:
            print(f"Global step:    {checkpoint['global_step']:,}")
        
        if 'best_val_loss' in checkpoint:
            val_loss = checkpoint['best_val_loss']
            perplexity = torch.exp(torch.tensor(val_loss)).item()
            print(f"Val loss:       {val_loss:.4f}")
            print(f"Perplexity:     {perplexity:.2f}")
        
        # Конфигурация модели
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"\nМодель:")
            print(f"  vocab_size:   {config.vocab_size}")
            print(f"  d_model:      {config.d_model}")
            print(f"  n_layers:     {config.n_layers}")
            print(f"  n_heads:      {config.n_heads}")
            print(f"  context_len:  {config.context_len}")
        
        # Размер файла
        file_size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)
        print(f"\nРазмер файла:   {file_size_mb:.1f} MB")
        
        return checkpoint.get('best_val_loss', float('inf'))
        
    except Exception as e:
        print(f"❌ Ошибка при чтении {checkpoint_path}: {e}")
        return float('inf')


def compare_all_checkpoints(checkpoint_dir: str = "checkpoints"):
    """Сравнивает все checkpoint'ы и находит лучший."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ Директория {checkpoint_dir} не найдена")
        return
    
    checkpoints = list(checkpoint_path.glob("*.pt"))
    
    if not checkpoints:
        print(f"❌ Нет checkpoint'ов в {checkpoint_dir}")
        return
    
    print(f"\n🔍 Найдено checkpoint'ов: {len(checkpoints)}\n")
    
    # Информация о каждом checkpoint
    results = []
    for cp in sorted(checkpoints):
        val_loss = inspect_checkpoint(str(cp))
        results.append((cp.name, val_loss))
    
    # Находим лучший
    best_checkpoint, best_loss = min(results, key=lambda x: x[1])
    
    print(f"\n{'='*60}")
    print(f"🏆 ЛУЧШИЙ CHECKPOINT")
    print(f"{'='*60}")
    print(f"Файл:      {best_checkpoint}")
    print(f"Val loss:  {best_loss:.4f}")
    print(f"\nИспользуйте для генерации:")
    print(f"  python inference.py --checkpoint checkpoints/{best_checkpoint}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Если указан конкретный файл
        inspect_checkpoint(sys.argv[1])
    else:
        # Сравниваем все
        compare_all_checkpoints()
