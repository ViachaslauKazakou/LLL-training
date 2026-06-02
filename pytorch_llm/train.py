"""
train.py — entry point для обучения модели

Запуск:
    python train.py                         # обучение с нуля
    python train.py --continue checkpoint   # продолжить обучение
"""

import argparse
import torch

from config import get_small_config, TrainingConfig
from data import load_data, prepare_sample_data
from model import GPTModel
from training import Trainer, continue_training


def main():
    parser = argparse.ArgumentParser(description="Обучение GPT модели")
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/sample.txt",
        help="Путь к файлу с данными (.txt или .json)"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["small", "medium", "base"],
        default="small",
        help="Размер модели"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Количество эпох"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Размер батча"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/mps/cuda/cpu)"
    )
    parser.add_argument(
        "--continue-from",
        type=str,
        default=None,
        help="Путь к checkpoint для продолжения обучения"
    )
    parser.add_argument(
        "--prepare-sample",
        action="store_true",
        help="Создать sample датасет и выйти"
    )
    parser.add_argument(
        "--include-topics",
        action="store_true",
        default=True,
        help="Для JSON: включать названия топиков (по умолчанию True)"
    )
    parser.add_argument(
        "--no-include-topics",
        dest="include_topics",
        action="store_false",
        help="Для JSON: не включать названия топиков"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping: остановка после N проверок без улучшения (default: 5)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model",
        help="Название модели для сохранения checkpoint'ов (default: model)"
    )
    
    args = parser.parse_args()
    
    # Определяем устройство
    from config import get_device
    if args.device == "auto":
        args.device = get_device()
        training_logger.info(f"Auto-detected device: {args.device}")
    
    # Создаём sample датасет если нужно
    if args.prepare_sample:
        prepare_sample_data(args.data)
        return
    
    # Конфигурация модели
    if args.config == "small":
        from config import get_small_config
        model_config = get_small_config()
    elif args.config == "medium":
        from config import get_medium_config
        model_config = get_medium_config()
    else:
        from config import get_base_config
        model_config = get_base_config()
    
    # Обновляем параметры из аргументов
    model_config.batch_size = args.batch_size
    model_config.learning_rate = args.lr
    
    # Конфигурация обучения
    train_config = TrainingConfig(
        n_epochs=args.epochs,
        data_path=args.data,
        device=args.device,
        patience=args.patience
    )
    
    # Загружаем данные
    training_logger.info("Загрузка данных...")
    
    # Определяем формат данных
    from data import detect_file_format
    file_format = detect_file_format(args.data)
    training_logger.info(f"Формат: {file_format}")
    
    train_loader, val_loader, tokenizer = load_data(
        args.data,
        model_config.context_len,
        model_config.batch_size,
        include_topics=args.include_topics  # Параметр для JSON
    )
    
    # Обновляем vocab_size из токенизатора
    model_config.vocab_size = tokenizer.vocab_size() if hasattr(tokenizer, 'vocab_size') else len(tokenizer.vocab)
    
    # Продолжение обучения или с нуля?
    if args.continue_from:
        continue_training(
            args.continue_from,
            train_loader,
            val_loader,
            train_config,
            additional_epochs=args.epochs,
            device=args.device,
            tokenizer=tokenizer,
            model_name=args.model_name
        )
    else:
        # Создаём модель
        training_logger.info("\nСоздание модели...")
        model = GPTModel(model_config)
        
        # Создаём trainer и запускаем обучение
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            train_config,
            device=args.device,
            tokenizer=tokenizer,
            model_name=args.model_name
        )
        
        trainer.train()
    
    training_logger.info("\n✅ Готово! Checkpoint сохранён в checkpoints/")


if __name__ == "__main__":
    main()
