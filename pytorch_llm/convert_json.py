#!/usr/bin/env python3
"""
convert_json.py — Конвертер JSON → TXT для обучения модели

Поддерживаемые форматы JSON:
1. Interview ({"items": [{"topic", "content"}]}) — вопросы для интервью
2. Tasks ({"tasks": [{"question", "solution", "answer"}]}) — задачи из OCR
3. Forum:
    - {"User": ..., "messages": {"Тема": ["msg1", ...]}}
    - {"title": "...", "messages": [{"author": "...", "content": "..."}, ...]}
    — форумные сообщения

Использование:
    python convert_json.py input.json                    # Вывод в консоль
    python convert_json.py input.json output.txt         # Сохранить в файл
    python convert_json.py input.json --auto             # Авто-имя: input_converted.txt
"""

from logger import data_logger

import sys
import argparse
from pathlib import Path
from data import detect_json_type, convert_interview_json_to_text, convert_tasks_json_to_text, convert_forum_json_to_text


def convert_json(input_path: str, output_path: str = None, auto_name: bool = False):
    """
    Конвертирует JSON в TXT.
    
    Args:
        input_path: Путь к входному JSON
        output_path: Путь к выходному TXT (опционально)
        auto_name: Автоматически сгенерировать имя выходного файла
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        data_logger.info(f"❌ Файл не найден: {input_path}")
        sys.exit(1)
    
    # Определяем тип JSON
    data_logger.info(f"🔍 Анализ файла: {input_file.name}")
    json_type = detect_json_type(str(input_file))
    data_logger.info(f"📋 Тип JSON: {json_type}")
    data_logger.info("")
    
    # Конвертируем
    text = None
    
    if json_type == 'interview':
        text = convert_interview_json_to_text(str(input_file))
    elif json_type == 'tasks':
        text = convert_tasks_json_to_text(str(input_file))
    elif json_type == 'forum':
        text = convert_forum_json_to_text(str(input_file), include_topics=True)
    else:
        data_logger.info(f"❌ Неизвестный формат JSON!")
        data_logger.info(f"   Поддерживаемые форматы:")
        data_logger.info(f"   - interview: {{'items': [{{'topic', 'content'}}]}}")
        data_logger.info(f"   - tasks: {{'tasks': [{{'question', 'solution', 'answer'}}]}}")
        data_logger.info(f"   - forum(dict): {{'User': ..., 'messages': {{'topic': [...]}}}}")
        data_logger.info(f"   - forum(list): {{'title': '...', 'messages': [{{'author','content'}}]}}")
        sys.exit(1)
    
    if not text or len(text) == 0:
        data_logger.info(f"❌ Ошибка конвертации: пустой результат")
        sys.exit(1)
    
    data_logger.info("")
    data_logger.info(f"✅ Конвертация успешна!")
    data_logger.info(f"   Длина текста: {len(text):,} символов")
    data_logger.info(f"   Строк: {text.count(chr(10)):,}")
    data_logger.info("")
    
    # Сохраняем или выводим
    if output_path:
        output_file = Path(output_path)
    elif auto_name:
        output_file = input_file.parent / f"{input_file.stem}_converted.txt"
    else:
        # Вывод в консоль
        data_logger.info("=" * 60)
        data_logger.info("РЕЗУЛЬТАТ КОНВЕРТАЦИИ:")
        data_logger.info("=" * 60)
        data_logger.info(text[:2000])  # Первые 2000 символов
        if len(text) > 2000:
            data_logger.info("...")
            data_logger.info(f"(Еще {len(text) - 2000:,} символов)")
        data_logger.info("")
        data_logger.info("💡 Для сохранения в файл используйте:")
        data_logger.info(f"   python convert_json.py {input_path} output.txt")
        data_logger.info(f"   python convert_json.py {input_path} --auto")
        return
    
    # Сохраняем в файл
    output_file.write_text(text, encoding='utf-8')
    data_logger.info(f"💾 Сохранено в: {output_file}")
    data_logger.info(f"   Размер файла: {output_file.stat().st_size / 1024:.1f} KB")
    data_logger.info("")
    data_logger.info(f"🚀 Готово к обучению!")
    data_logger.info(f"   В Streamlit UI выберите: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Конвертер JSON → TXT для обучения модели',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python convert_json.py interview.json                    # Вывод в консоль
  python convert_json.py interview.json output.txt         # Сохранить в файл
  python convert_json.py interview.json --auto             # Авто-имя
  python convert_json.py data/*.json --auto                # Конвертировать все JSON
        """
    )
    
    parser.add_argument(
        'input',
        help='Путь к входному JSON файлу'
    )
    
    parser.add_argument(
        'output',
        nargs='?',
        help='Путь к выходному TXT файлу (опционально)'
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Автоматически сгенерировать имя выходного файла (input_converted.txt)'
    )
    
    args = parser.parse_args()
    
    # Проверка аргументов
    if args.output and args.auto:
        data_logger.info("❌ Ошибка: нельзя использовать одновременно output и --auto")
        sys.exit(1)
    
    convert_json(args.input, args.output, args.auto)


if __name__ == '__main__':
    main()
