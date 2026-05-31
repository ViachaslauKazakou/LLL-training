#!/bin/bash

# Скрипт быстрого запуска приложений PyTorch LLM

set -e

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🚀 PyTorch LLM - Быстрый запуск${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Выберите приложение:"
echo ""
echo "  1) 🤖 Тренировка трансформера (Streamlit UI)"
echo "  2) 🔍 OCR распознавание (подготовка данных)"
echo "  3) 🚀 API сервер (OpenAI-совместимый)"
echo "  4) ⚙️  CLI тренировка (без UI)"
echo "  5) 💬 CLI inference (тестирование модели)"
echo ""
read -p "Введите номер [1-5]: " choice

case $choice in
    1)
        echo -e "${GREEN}▶ Запускаю Streamlit UI для тренировки...${NC}"
        poetry run streamlit run app.py --server.port 8501
        ;;
    2)
        echo -e "${GREEN}▶ Запускаю OCR приложение...${NC}"
        poetry run streamlit run ocr_app.py --server.port 8503
        ;;
    3)
        echo -e "${GREEN}▶ Запускаю API сервер на порту 8000...${NC}"
        poetry run uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
        ;;
    4)
        echo -e "${GREEN}▶ Запускаю CLI тренировку...${NC}"
        echo -e "${YELLOW}Используйте: poetry run python train.py${NC}"
        poetry run python train.py
        ;;
    5)
        echo -e "${GREEN}▶ Запускаю CLI inference...${NC}"
        echo -e "${YELLOW}Используйте: poetry run python inference.py${NC}"
        poetry run python inference.py
        ;;
    *)
        echo -e "${YELLOW}⚠️  Неверный выбор${NC}"
        exit 1
        ;;
esac
