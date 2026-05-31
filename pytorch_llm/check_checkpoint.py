import torch
import sys

checkpoint_path = "checkpoints/best_model.pt"

print(f"Загрузка checkpoint: {checkpoint_path}")
cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print("\n" + "="*60)
print("📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
print("="*60)

print(f"\n🎯 Метрики:")
print(f"  Global step: {cp.get('global_step', 'N/A')}")
val_loss = cp.get('best_val_loss', float('inf'))
if val_loss != float('inf'):
    print(f"  Best validation loss: {val_loss:.4f}")
    print(f"  Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")
else:
    print(f"  Best validation loss: N/A")

print(f"\n🏗️ Конфигурация модели:")
config = cp['config']
print(f"  Размер словаря: {config.vocab_size:,}")
print(f"  d_model: {config.d_model}")
print(f"  Слоёв: {config.n_layers}")
print(f"  Голов attention: {config.n_heads}")
print(f"  Длина контекста: {config.context_len}")
print(f"  Dropout: {config.dropout}")

# Подсчёт параметров (примерно)
embedding_params = config.vocab_size * config.d_model
pos_encoding_params = config.context_len * config.d_model
attention_params_per_layer = 4 * config.d_model * config.d_model  # Q, K, V, O
ff_params_per_layer = 2 * config.d_model * config.d_ff
layer_params = (attention_params_per_layer + ff_params_per_layer) * config.n_layers
total_params = embedding_params + pos_encoding_params + layer_params

print(f"\n📈 Параметры модели:")
print(f"  Всего параметров: ~{total_params:,}")
print(f"  Размер модели: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")

print("\n" + "="*60)
print("✅ Модель готова к использованию!")
print("="*60)

print(f"\n💡 Как использовать:")
print(f"  1. Запустите: streamlit run app.py")
print(f"  2. Перейдите во вкладку 'Генерация'")
print(f"  3. Выберите checkpoint: best_model.pt")
print(f"  4. Генерируйте текст!")
