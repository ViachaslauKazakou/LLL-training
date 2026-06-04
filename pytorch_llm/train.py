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
from logger import training_logger
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
        default=30,
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
        default=1.5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["hybrid", "bpe", "tiktoken", "char", "char_new"],
        default="hybrid",
        help="Тип токенизатора для обучения (по умолчанию: hybrid)"
    )
    parser.add_argument(
        "--tokenizer-encoding",
        type=str,
        default="cl100k_base",
        help="Имя tiktoken энкодера (используется только при --tokenizer tiktoken)"
    )
    parser.add_argument(
        "--hybrid-min-freq",
        type=int,
        default=2,
        help="Минимальная частота токена для HybridTokenizer"
    )
    parser.add_argument(
        "--hybrid-max-domain-tokens",
        type=int,
        default=8000,
        help="Максимум доменных токенов для HybridTokenizer"
    )
    parser.add_argument(
        "--bpe-vocab-size",
        type=int,
        default=8000,
        help="Целевой размер словаря для BPETokenizer (рекомендуемо 4000-16000)"
    )
    parser.add_argument(
        "--bpe-min-frequency",
        type=int,
        default=2,
        help="Минимальная частота merge для BPETokenizer"
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
        default=15,
        help="Early stopping: остановка после N проверок без улучшения (default: 15)"
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=20,
        help="Минимальное число эпох до возможного early stopping (default: 20)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Число шагов warmup learning rate (default: 200)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Шаг между окнами датасета. None = auto. Для overlap используйте context_len//2"
    )
    parser.add_argument(
        "--clean-forum",
        action="store_true",
        default=False,
        help="Очистить форумный шум (username#:, имена авторов) перед обучением"
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
        patience=args.patience,
        min_epochs=args.min_epochs,
        eval_every=150,
        warmup_steps=args.warmup_steps,
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
        include_topics=args.include_topics,  # Параметр для JSON
        tokenizer_type=args.tokenizer,
        tokenizer_encoding=args.tokenizer_encoding,
        hybrid_min_token_freq=args.hybrid_min_freq,
        hybrid_max_domain_tokens=args.hybrid_max_domain_tokens,
        bpe_vocab_size=args.bpe_vocab_size,
        bpe_min_frequency=args.bpe_min_frequency,
        stride=args.stride,
        clean_forum=args.clean_forum,
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
