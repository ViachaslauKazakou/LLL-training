#!/usr/bin/env python3
"""
Быстрый запуск приложений PyTorch LLM
"""

import subprocess
import sys
from pathlib import Path
from logger import app_logger

# Цвета для терминала
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
RESET = '\033[0m'

def print_header():
    app_logger.info(f"{BLUE}{'━' * 60}{RESET}")
    app_logger.info(f"{GREEN}🚀 PyTorch LLM - Быстрый запуск{RESET}")
    app_logger.info(f"{BLUE}{'━' * 60}{RESET}\n")

def print_menu():
    app_logger.info("Выберите приложение:\n")
    app_logger.info(f"  {CYAN}1){RESET} 🤖 Тренировка трансформера (Streamlit UI)")
    app_logger.info(f"  {CYAN}2){RESET} 🔍 OCR распознавание (подготовка данных)")
    app_logger.info(f"  {CYAN}3){RESET} 🚀 API сервер (OpenAI-совместимый)")
    app_logger.info(f"  {CYAN}4){RESET} ⚙️  CLI тренировка")
    app_logger.info(f"  {CYAN}5){RESET} 💬 CLI inference (тестирование)")
    app_logger.info(f"  {CYAN}6){RESET} 📊 Сравнить чекпоинты")
    app_logger.info(f"  {CYAN}0){RESET} ❌ Выход\n")

def run_command(cmd: list, description: str):
    app_logger.info(f"{GREEN}▶ {description}...{RESET}\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        app_logger.info(f"{YELLOW}⚠️  Ошибка при выполнении команды{RESET}")
        sys.exit(1)
    except KeyboardInterrupt:
        app_logger.info(f"\n{YELLOW}⚠️  Прервано пользователем{RESET}")
        sys.exit(0)

def main():
    # Проверяем что мы в правильной директории
    if not Path("app.py").exists():
        app_logger.info(f"{YELLOW}⚠️  Запустите скрипт из директории pytorch_llm/{RESET}")
        sys.exit(1)
    
    print_header()
    print_menu()
    
    try:
        choice = input("Введите номер [0-6]: ").strip()
    except KeyboardInterrupt:
        app_logger.info(f"\n{YELLOW}⚠️  Прервано{RESET}")
        sys.exit(0)
    
    if choice == "1":
        run_command(
            ["poetry", "run", "streamlit", "run", "app.py", "--server.port", "8501"],
            "Запускаю Streamlit UI для тренировки"
        )
    elif choice == "2":
        run_command(
            ["poetry", "run", "streamlit", "run", "ocr_app.py", "--server.port", "8503"],
            "Запускаю OCR приложение"
        )
    elif choice == "3":
        run_command(
            ["poetry", "run", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
            "Запускаю API сервер на порту 8000"
        )
    elif choice == "4":
        run_command(
            ["poetry", "run", "python", "train.py"],
            "Запускаю CLI тренировку"
        )
    elif choice == "5":
        run_command(
            ["poetry", "run", "python", "inference.py"],
            "Запускаю CLI inference"
        )
    elif choice == "6":
        run_command(
            ["poetry", "run", "python", "compare_checkpoints.py"],
            "Запускаю сравнение чекпоинтов"
        )
    elif choice == "0":
        app_logger.info(f"{GREEN}👋 До свидания!{RESET}")
        sys.exit(0)
    else:
        app_logger.info(f"{YELLOW}⚠️  Неверный выбор. Используйте числа от 0 до 6.{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
