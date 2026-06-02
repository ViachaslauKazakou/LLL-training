"""
example.py — простой пример обучения и генерации

Показывает полный пайплайн от подготовки данных до генерации текста.
"""

import torch
from pathlib import Path
from logger import training_logger, inference_logger

from config import get_small_config, TrainingConfig
from data import load_data, prepare_sample_data
from model import GPTModel
from training import Trainer
from inference import generate_text, CharTokenizer


def main():
    training_logger.info("="*60)
    training_logger.info("PyTorch LLM — пример использования")
    training_logger.info("="*60 + "\n")
    
    # 1. Подготовка данных
    training_logger.info("1. Подготовка данных...")
    data_path = "data/sample.txt"
    
    if not Path(data_path).exists():
        training_logger.info(f"   Создаём sample датасет: {data_path}")
        prepare_sample_data(data_path)
    
    # 2. Конфигурация
    training_logger.info("\n2. Создание конфигураций...")
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
    
    training_logger.info(f"   Устройство: {train_config.device}")
    training_logger.info(f"   Эпох: {train_config.n_epochs}")
    
    # 3. Загрузка данных
    training_logger.info("\n3. Загрузка данных...")
    train_loader, val_loader, tokenizer = load_data(
        data_path,
        model_config.context_len,
        model_config.batch_size
    )
    
    # Обновляем vocab_size из токенизатора
    model_config.vocab_size = len(tokenizer.vocab)
    training_logger.info(f"   Размер словаря: {model_config.vocab_size}")
    
    # 4. Создание модели
    training_logger.info("\n4. Создание модели...")
    model = GPTModel(model_config)
    training_logger.info(f"   Параметров: {model.count_parameters():,}")
    
    # 5. Обучение
    training_logger.info("\n5. Обучение...\n")
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        train_config,
        device=train_config.device
    )
    
    trainer.train()
    
    # 6. Генерация
    training_logger.info("\n6. Генерация текста...\n")
    
    prompts = [
        "Что такое вещество?",
        "Машинное обучение",
        "Глубокое обучение"
    ]
    
    model.eval()
    
    for prompt in prompts:
        training_logger.info(f"Промпт: {prompt}")
        training_logger.info("-" * 60)
        
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            device=train_config.device
        )
        
        training_logger.info(generated)
        training_logger.info("=" * 60 + "\n")
    
    training_logger.info("✅ Пример завершён!")
    training_logger.info(f"\nCheckpoint сохранён в: {train_config.checkpoint_dir}/")


if __name__ == "__main__":
    main()
