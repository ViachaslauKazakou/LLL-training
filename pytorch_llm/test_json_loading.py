"""Тест загрузки JSON данных."""
from data import convert_forum_json_to_text, load_data
from logger import data_logger

# Тест 1: Конвертация JSON -> текст
data_logger.info("=" * 60)
data_logger.info("ТЕСТ 1: Конвертация JSON -> текст")
data_logger.info("=" * 60)

text = convert_forum_json_to_text(
    'data/parser_2039898_20260521T071908.json',
    include_topics=True
)

data_logger.info(f"\nПример текста (первые 500 символов):")
data_logger.info(text[:500])
data_logger.info("...")

# Тест 2: Загрузка данных через load_data
data_logger.info("\n" + "=" * 60)
data_logger.info("ТЕСТ 2: Загрузка через load_data")
data_logger.info("=" * 60)

train_loader, val_loader, tokenizer = load_data(
    'data/parser_2039898_20260521T071908.json',
    context_len=128,
    batch_size=4,
    include_topics=True
)

data_logger.info(f"\nTrain batches: {len(train_loader)}")
data_logger.info(f"Val batches: {len(val_loader)}")
data_logger.info(f"Vocab size: {len(tokenizer.vocab)}")

# Тест 3: Пример батча
data_logger.info("\n" + "=" * 60)
data_logger.info("ТЕСТ 3: Пример батча")
data_logger.info("=" * 60)

batch = next(iter(train_loader))
input_ids, targets = batch

data_logger.info(f"Input shape: {input_ids.shape}")
data_logger.info(f"Targets shape: {targets.shape}")

# Декодируем первый пример
sample_text = tokenizer.decode(input_ids[0].tolist())
data_logger.info(f"\nПример входа (первые 200 символов):")
data_logger.info(sample_text[:200])
