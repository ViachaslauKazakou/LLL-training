"""
inference.py — генерация текста обученной моделью

Использование:
    python inference.py --checkpoint checkpoints/best_model.pt --prompt "Искусственный интеллект"
"""

import argparse
import torch
from typing import Union
import time

from model import GPTModel
from tokenizer import CharTokenizer, TikTokenizer
from data import prepare_sample_data
from pathlib import Path
from logger import inference_logger, log_session_start, log_session_end


def load_model_for_inference(checkpoint_path: str, device: str = "auto") -> tuple[GPTModel, dict]:
    """
    Загружает модель из checkpoint для генерации.
    
    Returns:
        model, checkpoint_dict
    """
    inference_logger.info(f"Загрузка checkpoint: {checkpoint_path}")
    
    # Определяем устройство
    from config import get_device
    if device == "auto":
        device = get_device()
    
    inference_logger.info(f"Используемое устройство: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['config']
    
    model = GPTModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    params = model.count_parameters()
    inference_logger.info(f"Модель загружена: {params:,} параметров")
    inference_logger.info(f"Global step: {checkpoint.get('global_step', 'N/A')}")
    inference_logger.info(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    inference_logger.info(f"✓ Модель загружена: {checkpoint_path}")
    inference_logger.info(f"  Параметров: {params:,}")
    
    return model, checkpoint


@torch.no_grad()
def generate_text(
    model: GPTModel,
    tokenizer: Union[CharTokenizer, TikTokenizer],
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda"
) -> tuple[str, dict]:
    """
    Генерирует текст на основе промпта.
    
    prompt: начальная строка
    max_new_tokens: сколько токенов сгенерировать
    temperature: креативность (выше = случайнее)
    top_k: выбирать из top-k наиболее вероятных токенов
    
    Returns:
        (generated_text, stats) где stats содержит:
            - total_time: общее время генерации (сек)
            - tokens_generated: количество сгенерированных токенов
            - tokens_per_second: скорость генерации (токенов/сек)
            - prompt_tokens: количество токенов в промпте
    """
    
    inference_logger.info(f"Генерация текста")
    inference_logger.info(f"Промпт: '{prompt}' ({len(prompt)} символов)")
    inference_logger.info(f"Параметры: max_tokens={max_new_tokens}, temp={temperature}, top_k={top_k}, top_p={top_p}")
    
    # Кодируем промпт
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device
    )
    prompt_tokens = input_ids.shape[1]
    
    # Замер времени генерации
    start_time = time.time()
    
    # Генерируем
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Декодируем
    generated = tokenizer.decode(output_ids[0].tolist())
    
    # Вычисляем статистику
    tokens_generated = output_ids.shape[1] - prompt_tokens
    tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
    
    stats = {
        'total_time': total_time,
        'tokens_generated': tokens_generated,
        'tokens_per_second': tokens_per_second,
        'prompt_tokens': prompt_tokens
    }
    
    inference_logger.info(f"Сгенерировано: {len(generated)} символов ({tokens_generated} токенов)")
    inference_logger.info(f"Скорость: {tokens_per_second:.2f} tokens/sec, время: {total_time:.2f} сек")
    
    return generated, stats


def interactive_generation(
    model: GPTModel,
    tokenizer: Union[CharTokenizer, TikTokenizer],
    device: str = "cuda"
):
    """Интерактивный режим генерации."""
    log_session_start(inference_logger, "Интерактивная генерация")
    
    inference_logger.info("\n" + "="*60)
    inference_logger.info("Интерактивная генерация")
    inference_logger.info("Введите 'q' для выхода")
    inference_logger.info("="*60 + "\n")
    
    while True:
        try:
            prompt = input("Промпт: ").strip()
            
            if prompt.lower() == 'q':
                break
            
            if not prompt:
                continue
            
            # Параметры генерации
            max_tokens = int(input("Максимум токенов [100]: ") or "100")
            temperature = float(input("Temperature [0.8]: ") or "0.8")
            top_k = int(input("Top-k [50]: ") or "50")
            top_p = float(input("Top-p [0.9]: ") or "0.9")

            inference_logger.info("\nГенерация...\n")

            generated, stats = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device
            )
            
            inference_logger.info("─" * 60)
            inference_logger.info(generated)
            inference_logger.info("─" * 60)
            inference_logger.info(f"⚡ Статистика:")
            inference_logger.info(f"  Время: {stats['total_time']:.2f} сек")
            inference_logger.info(f"  Скорость: {stats['tokens_per_second']:.1f} токенов/сек")
            inference_logger.info(f"  Сгенерировано: {stats['tokens_generated']} токенов")
            inference_logger.info(f"  Промпт: {stats['prompt_tokens']} токенов")
            inference_logger.info("─" * 60 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            inference_logger.info(f"Ошибка: {e}\n")
            inference_logger.error(f"Ошибка при генерации: {e}")
    
    log_session_end(inference_logger, "Интерактивная генерация")
    inference_logger.info("\nДо свидания!")


def main():
    parser = argparse.ArgumentParser(description="Генерация текста")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Путь к checkpoint модели"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sample.txt",
        help="Путь к данным (для токенизатора)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Начальный текст для генерации"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Максимум новых токенов"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature (выше = креативнее)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling (top-p). 0.9 = выбирать из токенов с суммарной вероятностью 90%%"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/mps/cuda/cpu)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Интерактивный режим"
    )
    
    args = parser.parse_args()
    
    # Загружаем модель
    model, checkpoint = load_model_for_inference(args.checkpoint, args.device)
    
    # Создаём токенизатор из checkpoint
    if 'tokenizer_config' in checkpoint:
        # Новый формат: tokenizer_config содержит тип и параметры
        tok_config = checkpoint['tokenizer_config']
        if tok_config['type'] == 'tiktoken':
            tokenizer = TikTokenizer.from_dict(tok_config)
            inference_logger.info(f"TikTokenizer загружен: {tokenizer.vocab_size()} токенов")
            inference_logger.info(f"✓ TikTokenizer загружен: {tokenizer.vocab_size()} токенов")
        else:
            tokenizer = CharTokenizer.from_dict(tok_config)
            inference_logger.info(f"CharTokenizer загружен: {tokenizer.vocab_size()} символов")
            inference_logger.info(f"✓ CharTokenizer загружен: {tokenizer.vocab_size()} символов")
    elif 'vocab' in checkpoint:
        # Старый формат: vocab напрямую (CharTokenizer)
        tokenizer = CharTokenizer.__new__(CharTokenizer)
        tokenizer.vocab = checkpoint['vocab']
        tokenizer.char_to_idx = {ch: idx for idx, ch in enumerate(tokenizer.vocab)}
        tokenizer.idx_to_char = {idx: ch for ch, idx in tokenizer.char_to_idx.items()}
        inference_logger.info(f"CharTokenizer загружен (legacy): {len(tokenizer.vocab)} символов")
        inference_logger.info(f"✓ CharTokenizer загружен (legacy): {len(tokenizer.vocab)} символов")
    else:
        # Fallback: создаём TikTokenizer по умолчанию
        tokenizer = TikTokenizer(encoding_name='cl100k_base')
        inference_logger.warning("Checkpoint без токенизатора, создан TikTokenizer (cl100k_base)")
        inference_logger.warning(f"⚠️  Checkpoint без токенизатора, создан TikTokenizer: {tokenizer.vocab_size()} токенов")
    
    # Интерактивный режим или разовая генерация?
    if args.interactive:
        interactive_generation(model, tokenizer, args.device)
    else:
        if args.prompt is None:
            inference_logger.info("Укажите --prompt или используйте --interactive")
            return
        
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
        
        inference_logger.info("\n" + "="*60)
        inference_logger.info(generated)
        inference_logger.info("="*60)


if __name__ == "__main__":
    main()
