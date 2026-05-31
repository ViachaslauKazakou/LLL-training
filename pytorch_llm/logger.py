"""
logger.py — система логирования для PyTorch LLM

Логирует операции в файлы:
- logs/data.log — загрузка данных
- logs/training.log — процесс обучения
- logs/inference.log — генерация текста
- logs/app.log — UI события
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Директория для логов
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Создаёт и настраивает logger с выводом в файл и консоль.
    
    Args:
        name: Имя логгера
        log_file: Имя файла для логов (относительно logs/)
        level: Уровень логирования
        format_string: Формат сообщений (по умолчанию timestamp + level + message)
    
    Returns:
        Настроенный logger
    """
    # Формат по умолчанию
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(message)s"
    
    # Создаём logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Удаляем существующие handlers (если есть)
    logger.handlers = []
    
    # Formatter
    formatter = logging.Formatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    log_path = LOGS_DIR / log_file
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (только для WARNING и выше, чтобы не дублировать print)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# ═══════════════════════════════════════════════════════════
# Готовые логгеры для разных модулей
# ═══════════════════════════════════════════════════════════

# Логгер для загрузки данных
data_logger = setup_logger(
    "pytorch_llm.data",
    "data.log",
    level=logging.INFO
)

# Логгер для обучения
training_logger = setup_logger(
    "pytorch_llm.training",
    "training.log",
    level=logging.INFO
)

# Логгер для генерации
inference_logger = setup_logger(
    "pytorch_llm.inference",
    "inference.log",
    level=logging.INFO
)

# Логгер для UI (Streamlit)
app_logger = setup_logger(
    "pytorch_llm.app",
    "app.log",
    level=logging.INFO
)


def log_session_start(logger: logging.Logger, module_name: str):
    """Логирует начало новой сессии работы модуля."""
    logger.info("=" * 60)
    logger.info(f"{module_name} — НАЧАЛО СЕССИИ")
    logger.info(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


def log_session_end(logger: logging.Logger, module_name: str):
    """Логирует завершение сессии модуля."""
    logger.info("=" * 60)
    logger.info(f"{module_name} — КОНЕЦ СЕССИИ")
    logger.info(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60 + "\n")


# Пример использования:
# from logger import data_logger, log_session_start, log_session_end
#
# log_session_start(data_logger, "Загрузка данных")
# data_logger.info("Загружен файл: sample.txt")
# data_logger.info("Токенов: 1000")
# log_session_end(data_logger, "Загрузка данных")
