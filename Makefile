.PHONY: help install cli ui train clean transformer ocr api inference pytorch tesseract

POETRY      := poetry run
V3_DIR      := mini_llm/v3
PYTORCH_DIR := pytorch_llm
CORPUS      := $(V3_DIR)/corpus.json
MODEL       := $(V3_DIR)/saved/model.npz

# ── Справка ─────────────────────────────────────────────────────────── #
help:
	@echo ""
	@echo "  LLM Learn — команды"
	@echo "  ────────────────────────────────────────────────────────────"
	@echo "  📦 Установка:"
	@echo "  make install       Установить зависимости (poetry)"
	@echo ""
	@echo "  🔸 PyTorch (главное):"
	@echo "  make pytorch       Запуск PyTorch трансформера"
	@echo "  make tesseract     Запуск OCR сканера"
	@echo ""
	@echo "  🔹 Mini LLM (NumPy):"
	@echo "  make cli           Запустить консольный интерфейс"
	@echo "  make ui            Запустить Streamlit UI"
	@echo "  make train         Обучить и сохранить модель"
	@echo ""
	@echo "  🔧 Дополнительно:"
	@echo "  make api           API сервер (OpenAI-совместимый)"
	@echo "  make inference     CLI тестирование модели"
	@echo ""
	@echo "  🧹 Очистка:"
	@echo "  make clean         Удалить модели и кэш"
	@echo "  ────────────────────────────────────────────────────────────"
	@echo ""

# ── Зависимости ─────────────────────────────────────────────────────── #
install:
	poetry install

# ── Консольный интерфейс ─────────────────────────────────────────────── #
cli:
	cd $(V3_DIR) && $(POETRY) python cli.py --corpus corpus.json --model saved/model.npz

# ── Streamlit UI ─────────────────────────────────────────────────────── #
ui:
	cd $(V3_DIR) && $(POETRY) streamlit run app.py

# ── Быстрое обучение и сохранение без интерактива ────────────────────── #
train:
	cd $(V3_DIR) && $(POETRY) python -c "\
from model import MiniLLM; \
llm = MiniLLM(); \
llm.load_corpus('corpus.json'); \
llm.train(n_epochs=500, lr=0.01); \
llm.save('saved/model.npz'); \
print('Done. Loss:', llm.train_history[-1])"

# ══════════════════════════════════════════════════════════════════════
# PyTorch Transformer
# ══════════════════════════════════════════════════════════════════════

# ── PyTorch трансформер (главная команда) ──────────────────────────────── #
pytorch:
	@echo "🤖 Запускаю PyTorch трансформер..."
	@echo "📍 http://localhost:8502"
	cd $(PYTORCH_DIR) && $(POETRY) streamlit run app.py --server.port 8502

# ── OCR сканер с Tesseract ─────────────────────────────────────────────── #
tesseract:
	@echo "🔍 Запускаю OCR сканер..."
	@echo "📍 http://localhost:8503"
	cd $(PYTORCH_DIR) && $(POETRY) streamlit run ocr_app.py --server.port 8503

# ── Алиасы (для совместимости) ─────────────────────────────────────────── #
transformer: pytorch
ocr: tesseract

# ── API сервер (OpenAI-совместимый) ────────────────────────────────────── #
api:
	@echo "🚀 Запускаю API сервер..."
	@echo "📍 http://localhost:8000"
	@echo "📍 http://localhost:8000/docs (Swagger UI)"
	cd $(PYTORCH_DIR) && $(POETRY) uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# ── CLI inference (тестирование модели) ────────────────────────────────── #
inference:
	@echo "💬 Запускаю CLI inference..."
	cd $(PYTORCH_DIR) && $(POETRY) python inference.py

# ── Очистка ──────────────────────────────────────────────────────────── #
clean:
	@echo "🧹 Очистка кэша и временных файлов..."
	rm -f $(MODEL)
	find mini_llm -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find $(PYTORCH_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find $(PYTORCH_DIR)/logs -type f -name "*.log" -mtime +7 -delete 2>/dev/null; true
	@echo "✅ Готово!"
