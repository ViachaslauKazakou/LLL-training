import torch
import sys
from logger import app_logger

checkpoint_path = "checkpoints/best_model.pt"

app_logger.info(f"Загрузка checkpoint: {checkpoint_path}")
cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

app_logger.info("\n" + "="*60)
app_logger.info("📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
app_logger.info("="*60)

app_logger.info(f"\n🎯 Метрики:")
app_logger.info(f"  Global step: {cp.get('global_step', 'N/A')}")
val_loss = cp.get('best_val_loss', float('inf'))
if val_loss != float('inf'):
    app_logger.info(f"  Best validation loss: {val_loss:.4f}")
    app_logger.info(f"  Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")
else:
    app_logger.info(f"  Best validation loss: N/A")

app_logger.info(f"\n🏗️ Конфигурация модели:")
config = cp['config']
app_logger.info(f"  Размер словаря: {config.vocab_size:,}")
app_logger.info(f"  d_model: {config.d_model}")
app_logger.info(f"  Слоёв: {config.n_layers}")
app_logger.info(f"  Голов attention: {config.n_heads}")
app_logger.info(f"  Длина контекста: {config.context_len}")
app_logger.info(f"  Dropout: {config.dropout}")

# Подсчёт параметров (примерно)
embedding_params = config.vocab_size * config.d_model
pos_encoding_params = config.context_len * config.d_model
attention_params_per_layer = 4 * config.d_model * config.d_model  # Q, K, V, O
ff_params_per_layer = 2 * config.d_model * config.d_ff
layer_params = (attention_params_per_layer + ff_params_per_layer) * config.n_layers
total_params = embedding_params + pos_encoding_params + layer_params

app_logger.info(f"\n📈 Параметры модели:")
app_logger.info(f"  Всего параметров: ~{total_params:,}")
app_logger.info(f"  Размер модели: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")

app_logger.info("\n" + "="*60)
app_logger.info("✅ Модель готова к использованию!")
app_logger.info("="*60)

app_logger.info(f"\n💡 Как использовать:")
app_logger.info(f"  1. Запустите: streamlit run app.py")
app_logger.info(f"  2. Перейдите во вкладку 'Генерация'")
app_logger.info(f"  3. Выберите checkpoint: best_model.pt")
app_logger.info(f"  4. Генерируйте текст!")
