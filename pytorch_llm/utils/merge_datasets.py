#!/usr/bin/env python3
"""
Объединитель датасетов и анализатор текстовых данных

Использование:
    python merge_datasets.py file1.txt file2.txt --output combined.txt
    python merge_datasets.py --dir data_folder/ --output combined.txt
    python merge_datasets.py --analyze combined.txt
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
import re

# Импорт логгера
try:
    from logger import data_logger
except ImportError:
    # Fallback если запускается из utils/
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from logger import data_logger


def analyze_dataset(file_path: Union[str, Path]) -> Dict:
    """
    Анализирует текстовый датасет.
    
    Args:
        file_path: Путь к файлу (строка или Path)
        
    Returns:
        Словарь со статистикой
    """
    file_path = Path(file_path)
    text = file_path.read_text(encoding='utf-8')
    
    # Базовая статистика
    lines = text.split('\n')
    words = text.split()
    chars = len(text)
    
    # Поиск кириллицы
    cyrillic_chars = len(re.findall(r'[а-яА-ЯёЁ]', text))
    cyrillic_lines = sum(1 for line in lines if re.search(r'[а-яА-ЯёЁ]', line))
    
    # Поиск латиницы
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    latin_lines = sum(1 for line in lines if re.search(r'[a-zA-Z]', line))
    
    # Пустые строки
    empty_lines = sum(1 for line in lines if not line.strip())
    
    stats = {
        'file_path': str(file_path),
        'size_bytes': file_path.stat().st_size,
        'size_kb': file_path.stat().st_size / 1024,
        'size_mb': file_path.stat().st_size / (1024 * 1024),
        'total_chars': chars,
        'total_words': len(words),
        'total_lines': len(lines),
        'empty_lines': empty_lines,
        'cyrillic_chars': cyrillic_chars,
        'cyrillic_lines': cyrillic_lines,
        'cyrillic_percent': (cyrillic_lines / len(lines) * 100) if lines else 0,
        'latin_chars': latin_chars,
        'latin_lines': latin_lines,
        'latin_percent': (latin_lines / len(lines) * 100) if lines else 0,
    }
    
    return stats


def print_analysis(stats: Dict):
    """Логирует анализ датасета"""
    data_logger.info(f"{'='*60}")
    data_logger.info(f"📊 АНАЛИЗ ДАТАСЕТА: {Path(stats['file_path']).name}")
    data_logger.info(f"{'='*60}")
    data_logger.info(f"📏 Размер: {stats['size_mb']:.2f} MB ({stats['size_kb']:.2f} KB)")
    data_logger.info(f"📝 Контент: {stats['total_chars']:,} символов, {stats['total_words']:,} слов, {stats['total_lines']:,} строк")
    data_logger.info(f"🌐 Язык: Кириллица {stats['cyrillic_percent']:.1f}%, Латиница {stats['latin_percent']:.1f}%")
    data_logger.info(f"{'='*60}")


def merge_text_files(file_paths: List[Union[str, Path]], output_path: Union[str, Path], 
                     separator: str = "\n\n---\n\n", 
                     add_headers: bool = True,
                     analyze: bool = True) -> Dict:
    """
    Объединяет несколько текстовых файлов в один.
    
    Args:
        file_paths: Список путей к файлам (строки или Path)
        output_path: Путь для сохранения объединенного файла (строка или Path)
        separator: Разделитель между файлами
        add_headers: Добавлять заголовки с именами файлов
        analyze: Показать анализ до и после
        
    Returns:
        Словарь с output_path и analysis (если analyze=True)
    """
    # Конвертируем в Path объекты
    file_paths = [Path(p) for p in file_paths]
    output_path = Path(output_path)
    
    if not file_paths:
        data_logger.warning("❌ Нет файлов для объединения")
        return {'output_path': None, 'analysis': None}
    
    data_logger.info(f"📚 Объединение {len(file_paths)} файлов")
    
    # Анализ исходных файлов
    if analyze:
        data_logger.info(f"{'='*60}")
        data_logger.info("📊 АНАЛИЗ ИСХОДНЫХ ФАЙЛОВ:")
        data_logger.info(f"{'='*60}")
        
        total_size = 0
        for i, file_path in enumerate(file_paths, 1):
            if not file_path.exists():
                data_logger.warning(f"⚠️  {i}. {file_path.name} - НЕ НАЙДЕН")
                continue
            
            size_kb = file_path.stat().st_size / 1024
            total_size += size_kb
            lines = len(file_path.read_text(encoding='utf-8').split('\n'))
            data_logger.info(f"  {i}. {file_path.name:<40} {size_kb:>8.1f} KB, {lines:>6,} строк")
        
        data_logger.info(f"  {'Итого:':<40} {total_size:>8.1f} KB")
        data_logger.info(f"{'='*60}")
    
    # Объединение
    combined_parts = []
    
    for file_path in file_paths:
        if not file_path.exists():
            data_logger.warning(f"⚠️  Пропуск: {file_path.name} (не найден)")
            continue
        
        text = file_path.read_text(encoding='utf-8')
        
        if add_headers:
            header = f"# ИСТОЧНИК: {file_path.name}\n\n"
            combined_parts.append(header + text)
        else:
            combined_parts.append(text)
    
    combined_text = separator.join(combined_parts)
    
    # Сохранение
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined_text, encoding='utf-8')
    
    data_logger.info(f"✅ Объединенный файл сохранен: {output_path}")
    
    # Анализ результата
    stats = None
    if analyze:
        stats = analyze_dataset(output_path)
        print_analysis(stats)
    
    return {
        'output_path': str(output_path),
        'analysis': stats
    }


def main():
    parser = argparse.ArgumentParser(
        description="Объединитель датасетов и анализатор",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Объединить конкретные файлы
  python merge_datasets.py file1.txt file2.txt file3.txt --output combined.txt
  
  # Объединить все файлы из директории
  python merge_datasets.py --dir data/ --output combined.txt --pattern "*.txt"
  
  # Только анализ
  python merge_datasets.py --analyze dataset.txt
  
  # Без заголовков и с кастомным разделителем
  python merge_datasets.py file1.txt file2.txt --output out.txt --no-headers --separator "\\n\\n"
        """
    )
    
    parser.add_argument('files', nargs='*', type=str,
                        help='Входные текстовые файлы')
    parser.add_argument('--dir', type=str,
                        help='Директория с текстовыми файлами')
    parser.add_argument('--pattern', type=str, default='*.txt',
                        help='Паттерн для поиска файлов (default: *.txt)')
    parser.add_argument('--output', '-o', type=str,
                        help='Выходной файл')
    parser.add_argument('--separator', type=str, default='\n\n---\n\n',
                        help='Разделитель между файлами (default: \\n\\n---\\n\\n)')
    parser.add_argument('--no-headers', action='store_true',
                        help='Не добавлять заголовки с именами файлов')
    parser.add_argument('--analyze', type=str,
                        help='Анализировать указанный файл')
    parser.add_argument('--no-analyze', action='store_true',
                        help='Не показывать анализ при объединении')
    
    args = parser.parse_args()
    
    # Режим анализа
    if args.analyze:
        file_path = Path(args.analyze)
        if not file_path.exists():
            data_logger.error(f"❌ Файл не существует: {file_path}")
            return
        
        stats = analyze_dataset(file_path)
        print_analysis(stats)
        return
    
    # Сбор файлов
    file_paths = []
    
    if args.dir:
        # Из директории
        dir_path = Path(args.dir)
        if not dir_path.exists():
            data_logger.error(f"❌ Директория не существует: {dir_path}")
            return
        
        file_paths = sorted(dir_path.glob(args.pattern))
    
    elif args.files:
        # Из списка
        file_paths = [Path(f) for f in args.files]
    
    else:
        parser.print_help()
        return
    
    # Проверка выходного файла
    if not args.output:
        data_logger.error("❌ Укажите выходной файл через --output")
        return
    
    output_path = Path(args.output)
    
    # Объединение
    merge_text_files(
        file_paths,
        output_path,
        separator=args.separator,
        add_headers=not args.no_headers,
        analyze=not args.no_analyze
    )


if __name__ == '__main__':
    main()
