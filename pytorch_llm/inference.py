"""
inference.py — генерация текста обученной моделью

Использование:
    python inference.py --checkpoint checkpoints/best_model.pt --prompt "Искусственный интеллект"
"""

import argparse
import torch

from model import GPTModel
from data import CharTokenizer, prepare_sample_data
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
    
    print(f"✓ Модель загружена: {checkpoint_path}")
    print(f"  Параметров: {params:,}")
    
    return model, checkpoint


@torch.no_grad()
def generate_text(
    model: GPTModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cuda"
) -> str:
    """
    Генерирует текст на основе промпта.
    
    prompt: начальная строка
    max_new_tokens: сколько токенов сгенерировать
    temperature: креативность (выше = случайнее)
    top_k: выбирать из top-k наиболее вероятных токенов
    """
    inference_logger.info(f"Генерация текста")
    inference_logger.info(f"Промпт: '{prompt}' ({len(prompt)} символов)")
    inference_logger.info(f"Параметры: max_tokens={max_new_tokens}, temp={temperature}, top_k={top_k}")
    
    # Кодируем промпт
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device
    )
    
    # Генерируем
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    
    # Декодируем
    generated = tokenizer.decode(output_ids[0].tolist())
    
    inference_logger.info(f"Сгенерировано: {len(generated)} символов")
    
    return generated


def interactive_generation(
    model: GPTModel,
    tokenizer: CharTokenizer,
    device: str = "cuda"
):
    """Интерактивный режим генерации."""
    log_session_start(inference_logger, "Интерактивная генерация")
    
    print("\n" + "="*60)
    print("Интерактивная генерация")
    print("Введите 'q' для выхода")
    print("="*60 + "\n")
    
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
            
            print("\nГенерация...\n")
            
            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            
            print("─" * 60)
            print(generated)
            print("─" * 60 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {e}\n")
            inference_logger.error(f"Ошибка при генерации: {e}")
    
    log_session_end(inference_logger, "Интерактивная генерация")
    print("\nДо свидания!")


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
    
    # Создаём токенизатор из checkpoint (если vocab есть) или из данных
    if 'vocab' in checkpoint:
        # Vocab сохранён в checkpoint - создаём токенизатор из него
        tokenizer = CharTokenizer.__new__(CharTokenizer)
        tokenizer.vocab = checkpoint['vocab']
        tokenizer.char_to_idx = {ch: idx for idx, ch in enumerate(tokenizer.vocab)}
        tokenizer.idx_to_char = {idx: ch for ch, idx in tokenizer.char_to_idx.items()}
        inference_logger.info(f"Токенизатор загружен из checkpoint: {len(tokenizer.vocab)} символов")
        print(f"✓ Токенизатор загружен из checkpoint: {len(tokenizer.vocab)} символов")
    else:
        # Старый checkpoint без vocab - создаём из файла данных
        text = Path(args.data).read_text(encoding='utf-8')
        tokenizer = CharTokenizer(text)
        inference_logger.warning("Checkpoint без vocab, создан токенизатор из файла данных")
        print(f"⚠️  Checkpoint без vocab, создан токенизатор из данных: {len(tokenizer.vocab)} символов")
    
    # Интерактивный режим или разовая генерация?
    if args.interactive:
        interactive_generation(model, tokenizer, args.device)
    else:
        if args.prompt is None:
            print("Укажите --prompt или используйте --interactive")
            return
        
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )
        
        print("\n" + "="*60)
        print(generated)
        print("="*60)


if __name__ == "__main__":
    main()
