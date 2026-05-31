"""Тест загрузки JSON данных."""
from data import convert_forum_json_to_text, load_data

# Тест 1: Конвертация JSON -> текст
print("=" * 60)
print("ТЕСТ 1: Конвертация JSON -> текст")
print("=" * 60)

text = convert_forum_json_to_text(
    'data/parser_2039898_20260521T071908.json',
    include_topics=True
)

print(f"\nПример текста (первые 500 символов):")
print(text[:500])
print("...")

# Тест 2: Загрузка данных через load_data
print("\n" + "=" * 60)
print("ТЕСТ 2: Загрузка через load_data")
print("=" * 60)

train_loader, val_loader, tokenizer = load_data(
    'data/parser_2039898_20260521T071908.json',
    context_len=128,
    batch_size=4,
    include_topics=True
)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Vocab size: {len(tokenizer.vocab)}")

# Тест 3: Пример батча
print("\n" + "=" * 60)
print("ТЕСТ 3: Пример батча")
print("=" * 60)

batch = next(iter(train_loader))
input_ids, targets = batch

print(f"Input shape: {input_ids.shape}")
print(f"Targets shape: {targets.shape}")

# Декодируем первый пример
sample_text = tokenizer.decode(input_ids[0].tolist())
print(f"\nПример входа (первые 200 символов):")
print(sample_text[:200])
