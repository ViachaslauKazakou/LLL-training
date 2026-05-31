"""
example.py — простой пример обучения и генерации

Показывает полный пайплайн от подготовки данных до генерации текста.
"""

import torch
from pathlib import Path

from config import get_small_config, TrainingConfig
from data import load_data, prepare_sample_data
from model import GPTModel
from training import Trainer
from inference import generate_text, CharTokenizer


def main():
    print("="*60)
    print("PyTorch LLM — пример использования")
    print("="*60 + "\n")
    
    # 1. Подготовка данных
    print("1. Подготовка данных...")
    data_path = "data/sample.txt"
    
    if not Path(data_path).exists():
        print(f"   Создаём sample датасет: {data_path}")
        prepare_sample_data(data_path)
    
    # 2. Конфигурация
    print("\n2. Создание конфигураций...")
    model_config = get_small_config()
    
    from config import get_device
    device = get_device()
    
    train_config = TrainingConfig(
        n_epochs=3,             # быстрый пример
        eval_every=100,
        save_every=500,
        data_path=data_path,
        device=device
    )
    
    print(f"   Устройство: {train_config.device}")
    print(f"   Эпох: {train_config.n_epochs}")
    
    # 3. Загрузка данных
    print("\n3. Загрузка данных...")
    train_loader, val_loader, tokenizer = load_data(
        data_path,
        model_config.context_len,
        model_config.batch_size
    )
    
    # Обновляем vocab_size из токенизатора
    model_config.vocab_size = len(tokenizer.vocab)
    print(f"   Размер словаря: {model_config.vocab_size}")
    
    # 4. Создание модели
    print("\n4. Создание модели...")
    model = GPTModel(model_config)
    print(f"   Параметров: {model.count_parameters():,}")
    
    # 5. Обучение
    print("\n5. Обучение...\n")
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        train_config,
        device=train_config.device
    )
    
    trainer.train()
    
    # 6. Генерация
    print("\n6. Генерация текста...\n")
    
    prompts = [
        "Искусственный интеллект",
        "Машинное обучение",
        "Глубокое обучение"
    ]
    
    model.eval()
    
    for prompt in prompts:
        print(f"Промпт: {prompt}")
        print("-" * 60)
        
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            device=train_config.device
        )
        
        print(generated)
        print("=" * 60 + "\n")
    
    print("✅ Пример завершён!")
    print(f"\nCheckpoint сохранён в: {train_config.checkpoint_dir}/")


if __name__ == "__main__":
    main()
